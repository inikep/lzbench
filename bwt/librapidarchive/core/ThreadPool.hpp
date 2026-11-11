#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <thread>
#include <type_traits>
#include <utility>
#include <unordered_map>
#include <vector>

#include "AffinityHelpers.hpp"
#include "JoiningThread.hpp"

#ifdef WITH_PYTHON_SUPPORT
    #include "ScopedGIL.hpp"
#endif


namespace rapidgzip
{
/**
 * Function evaluations can be given to a ThreadPool instance,
 * which assigns the evaluation to one of its threads to be evaluated in parallel.
 */
class ThreadPool
{
public:
    using ThreadPinning = std::unordered_map</* thread Index */ size_t, /* core ID */ uint32_t>;

private:
    /**
     * A small type-erasure function wrapper for non-copyable function objects with no function arguments.
     *
     * std::function<void()> won't work to wrap std::packaged_task
     * because the former requires a copy-constructible object, which the latter is not.
     * @see http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#1287
     *
     * By upcasting from a templated specialized derived class to this base class, the template type is "erased".
     * However, @ref BaseFunctor can't be directly used with very much correctly because the default
     * operations like copy, move and so on won't work correctly with the members in the derived class when it
     * is called from the base class type.
     * For this reason, this FunctionWrapper creates a unique_ptr of @ref BaseFunctor.
     */
    class PackagedTaskWrapper
    {
    private:
        struct BaseFunctor
        {
            virtual void
            operator()() = 0;

            virtual
            ~BaseFunctor() = default;
        };

        template<class T_Functor>
        struct SpecializedFunctor :
            BaseFunctor
        {
            explicit
            SpecializedFunctor( T_Functor&& functor ) :
                m_functor( std::move( functor ) )
            {}

            void
            operator()() override
            {
                m_functor();
            }

        private:
            T_Functor m_functor;
        };

    public:
        template<class T_Functor, std::enable_if_t<std::is_invocable_v<T_Functor>, void>* = nullptr>
        explicit
        PackagedTaskWrapper( T_Functor&& functor ) :
            m_impl( std::make_unique<SpecializedFunctor<T_Functor> >( std::forward<T_Functor>( functor ) ) )
        {}

        void
        operator()()
        {
            ( *m_impl )();
        }

    private:
        std::unique_ptr<BaseFunctor> m_impl;
    };

public:
    explicit
    ThreadPool( size_t        threadCount = availableCores(),
                ThreadPinning threadPinning = {} ) :
        m_threadCount( threadCount ),
        m_threadPinning( std::move( threadPinning ) )
    {
        m_threads.reserve( m_threadCount );
    }

    ~ThreadPool()
    {
        stop();
    }

    void
    stop()
    {
        {
            const std::lock_guard lock( m_mutex );
            m_threadPoolRunning = false;
            m_pingWorkers.notify_all();
        }

    #ifdef WITH_PYTHON_SUPPORT
        /* The GIL needs to be unlocked for the worker threads to not wait infinitely when calling methods
         * on a given Python file object. In this waiting time, they also wouldn't check m_threadPoolRunning,
         * nor PyErr_CheckSignals and therefore would deadlock. */
        const ScopedGILUnlock unlockedGIL;
    #endif

        m_threads.clear();
    }

    /**
     * Any function taking no arguments and returning any argument may be submitted to be executed.
     * The returned future can be used to access the result when it is really needed.
     * @param priority Tasks are processed ordered by their priority, i.e., tasks with priority 0
     *        are processed only after all tasks with priority 0 have been processed.
     */
    template<class T_Functor, std::enable_if_t<std::is_invocable_v<T_Functor>, void>* = nullptr>
    std::future<decltype( std::declval<T_Functor>()() )>
    submit( T_Functor&& task,
            int         priority = 0 )
    {
        const std::lock_guard lock( m_mutex );

        if ( m_threadCount == 0 ) {
            return std::async( std::launch::deferred, std::forward<T_Functor>( task ) );
        }

        /* A previous stop() may have joined all workers, e.g., before forking. Start them again on demand. */
        m_threadPoolRunning = true;

        /* Use a packaged task, which abstracts handling the return type and makes the task return void. */
        using ReturnType = decltype( std::declval<T_Functor>()() );
        std::packaged_task<ReturnType()> packagedTask{ std::forward<T_Functor>( task ) };
        auto resultFuture = packagedTask.get_future();
        m_tasks[priority].emplace_back( std::move( packagedTask ) );

        if ( ( m_threads.size() < m_threadCount ) && ( m_idleThreadCount == 0 ) ) {
            spawnThread();
        }

        m_pingWorkers.notify_one();

        return resultFuture;
    }

    [[nodiscard]] size_t
    capacity() const
    {
        return m_threadCount;
    }

    [[nodiscard]] size_t
    unprocessedTasksCount( const std::optional<int> priority = {} ) const
    {
        const std::lock_guard lock( m_mutex );
        if ( priority ) {
            const auto tasks = m_tasks.find( *priority );
            return tasks == m_tasks.end() ? 0 : tasks->second.size();
        }
        return std::accumulate( m_tasks.begin(), m_tasks.end(), size_t( 0 ),
                                [] ( size_t sum, const auto& tasks ) { return sum + tasks.second.size(); } );
    }

private:
    /**
     * Does not lock! Therefore it is a private method that should only be called with a lock.
     */
    [[nodiscard]] bool
    hasUnprocessedTasks() const
    {
        return std::any_of( m_tasks.begin(), m_tasks.end(),
                            [] ( const auto& tasks ) { return !tasks.second.empty(); } );
    }

    void
    workerMain( size_t threadIndex )
    {
        if ( const auto pinning = m_threadPinning.find( threadIndex ); pinning != m_threadPinning.end() ) {
            pinThreadToLogicalCore( static_cast<int>( pinning->second ) );
        }

        while ( m_threadPoolRunning )
        {
            std::unique_lock<std::mutex> tasksLock( m_mutex );
            ++m_idleThreadCount;
            m_pingWorkers.wait( tasksLock, [this] () { return hasUnprocessedTasks() || !m_threadPoolRunning; } );
            --m_idleThreadCount;

            if ( !m_threadPoolRunning ) {
                break;
            }

            const auto nonEmptyTasks = std::find_if( m_tasks.begin(), m_tasks.end(),
                                                     [] ( const auto& tasks ) { return !tasks.second.empty(); } );
            if ( nonEmptyTasks != m_tasks.end() ) {
                auto task = std::move( nonEmptyTasks->second.front() );
                nonEmptyTasks->second.pop_front();
                tasksLock.unlock();
                task();
            }
        }
    }

    void
    spawnThread()
    {
        m_threads.emplace_back( [this, i = m_threads.size()] () { workerMain( i ); } );
    }

private:
    std::atomic<bool> m_threadPoolRunning = true;

    const size_t m_threadCount;
    const ThreadPinning m_threadPinning;
    std::atomic<size_t> m_idleThreadCount{ 0 };

    std::map</* priority */ int, std::deque<PackagedTaskWrapper> > m_tasks;
    /** necessary for m_tasks AND m_pingWorkers or else the notify_all might go unnoticed! */

    mutable std::mutex m_mutex;
    std::condition_variable m_pingWorkers;

    /**
     * Should come last so that it's lifetime is the shortest, i.e., there is no danger for the other
     * members to not yet be constructed or be already destructed while a task is still running.
     */
    std::vector<JoiningThread> m_threads;
};
}  // namespace rapidgzip
