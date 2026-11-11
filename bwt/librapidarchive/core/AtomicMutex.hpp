#pragma once

#include <atomic>
#include <chrono>
#include <thread>

namespace rapidgzip
{
class AtomicMutex
{
public:
    void
    lock()
    {
        /* Wait until we successfully have set the flag ourselves.
         * Test with relaxed memory order first to improve speed because the modifying exchange has to
         * lock the cache line. */
        while ( m_flag.load( std::memory_order_relaxed ) || m_flag.exchange( true, std::memory_order_acquire ) ) {
            using namespace std::chrono_literals;
            std::this_thread::sleep_for( 10ns );
        }
    }

    void
    unlock()
    {
        m_flag.store( false, std::memory_order_release );
    }

private:
    std::atomic<bool> m_flag{ false };
};
}  // namespace rapidgzip
