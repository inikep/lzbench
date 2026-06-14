#include <gtest/gtest.h>
#include <cstring>
#include <vector>
#include <tuple>

// Import the actual function from the file under test
#include "bench/lz_codecs.cpp"

struct MemcpyTestCase {
    size_t insize;
    size_t outsize;
    std::string description;
};

class LzbenchMemcpySecurityTest : public ::testing::TestWithParam<MemcpyTestCase> {};

TEST_P(LzbenchMemcpySecurityTest, BufferReadNeverExceedsOutsize) {
    // Invariant: bytes written to outbuf must never exceed outsize
    auto tc = GetParam();

    std::vector<char> inbuf(tc.insize, 'A');
    std::vector<char> outbuf(tc.outsize, '\0');
    // Guard zone to detect overflow
    const size_t guard_size = 64;
    std::vector<char> guarded_out(tc.outsize + guard_size, '\0');
    std::memset(guarded_out.data() + tc.outsize, 0xDE, guard_size);

    int64_t result = lzbench_memcpy(inbuf.data(), tc.insize, guarded_out.data(), tc.outsize, nullptr);

    // The function must not write beyond outsize bytes
    // Check guard zone is untouched
    for (size_t i = 0; i < guard_size; i++) {
        EXPECT_EQ((unsigned char)guarded_out[tc.outsize + i], 0xDE)
            << "Buffer overflow detected at guard offset " << i
            << " with insize=" << tc.insize << " outsize=" << tc.outsize;
    }
    // Result should not exceed outsize
    EXPECT_LE((size_t)result, tc.outsize)
        << "Return value exceeds output buffer size: " << tc.description;
}

INSTANTIATE_TEST_SUITE_P(
    AdversarialInputs,
    LzbenchMemcpySecurityTest,
    ::testing::Values(
        MemcpyTestCase{2048, 1024, "insize 2x outsize - exploit case"},
        MemcpyTestCase{10240, 1024, "insize 10x outsize - extreme overflow"},
        MemcpyTestCase{1025, 1024, "insize exceeds outsize by 1 - boundary"},
        MemcpyTestCase{512, 1024, "valid case - insize < outsize"}
    )
);

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}