#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <utility>


namespace indexed_bzip2
{
/**
 * Accumulates the bzip2 stream CRC over data blocks decoded in stream order.
 *
 * The stream CRC is defined by the bzip2 format as a rotate-left-xor over the per-block CRCs. It can only be
 * verified when every data block of a stream was seen in order. Skipping a block, for example by seeking with
 * imported block offsets, invalidates the accumulator so later EOS blocks are not rejected because of a partial
 * accumulator. finishStream() verifies and resets the state for a following concatenated stream.
 */
class StreamCRCVerifier
{
public:
    void
    appendBlock( const size_t   decodedOffsetInBytes,
                 const size_t   decodedSizeInBytes,
                 const uint32_t blockCRC )
    {
        if ( !m_calculatedCRC.has_value() ) {
            return;
        }

        if ( decodedOffsetInBytes == m_nextDecodedOffset ) {
            *m_calculatedCRC = ( ( *m_calculatedCRC << 1U ) | ( *m_calculatedCRC >> 31U ) ) ^ blockCRC;
            m_nextDecodedOffset += decodedSizeInBytes;
        } else if ( decodedOffsetInBytes > m_nextDecodedOffset ) {
            /* A preceding block was skipped, so the stream CRC cannot be checked reliably. */
            m_calculatedCRC.reset();
        }
    }

    void
    finishStream( const size_t   eosOffsetInBits,
                  const size_t   decodedOffsetInBytes,
                  const uint32_t expectedCRC )
    {
        if ( m_lastHandledEOSOffset == eosOffsetInBits ) {
            return;
        }
        m_lastHandledEOSOffset = eosOffsetInBits;

        if ( m_calculatedCRC.has_value()
             && ( m_nextDecodedOffset == decodedOffsetInBytes )
             && ( *m_calculatedCRC != expectedCRC ) ) {
            std::stringstream message;
            message << "[BZip2] Stream CRC 0x" << std::hex << expectedCRC
                    << " does not match calculated CRC 0x" << *m_calculatedCRC;
            throw std::runtime_error( std::move( message ).str() );
        }

        m_calculatedCRC = 0U;
        m_nextDecodedOffset = decodedOffsetInBytes;
    }

private:
    /** std::nullopt means a block was skipped, so the stream CRC cannot be verified. */
    std::optional<uint32_t> m_calculatedCRC{ 0U };
    size_t m_nextDecodedOffset{ 0 };
    std::optional<size_t> m_lastHandledEOSOffset;
};
}  // namespace indexed_bzip2
