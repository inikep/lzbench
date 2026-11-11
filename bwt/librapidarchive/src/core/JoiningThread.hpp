#pragma once

#include <thread>
#include <utility>


namespace rapidgzip
{
/**
 * Similar to the planned C++20 std::jthread, this class joins in the destructor.
 * Additionally, it ensures that all threads created with this interface correctly initialize rpmalloc!
 */
class JoiningThread
{
public:
    template<class Function, class... Args>
    explicit
    JoiningThread( Function&& function, Args&&... args ) :
        m_thread( std::forward<Function>( function ), std::forward<Args>( args )... )
    {}

    JoiningThread( JoiningThread&& ) = default;

    JoiningThread( const JoiningThread& ) = delete;

    JoiningThread&
    operator=( JoiningThread&& ) = delete;

    JoiningThread&
    operator=( const JoiningThread& ) = delete;

    ~JoiningThread()
    {
        if ( m_thread.joinable() ) {
            m_thread.join();
        }
    }

    [[nodiscard]] std::thread::id
    get_id() const noexcept
    {
        return m_thread.get_id();
    }

    [[nodiscard]] bool
    joinable() const
    {
        return m_thread.joinable();
    }

    void
    join()
    {
        m_thread.join();
    }

private:
    std::thread m_thread;
};
}  // namespace rapidgzip
