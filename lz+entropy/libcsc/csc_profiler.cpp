#include <csc_profiler.h>
#ifdef _HAVE_PROFILER_
#include <stdio.h>
#include <string>
#include <vector>

using namespace std;



uint32_t matchlen_cnt[32] = {0};
uint32_t repmatch_idx_cnt[5] = {0};
uint32_t lits_cnt = 0;
uint32_t match1b_cnt = 0;
string lits;


struct DistMatch {
    uint32_t match_len;
    uint32_t match_dist;
};

void PEncodeLiteral(uint32_t c)
{
    lits_cnt++;

    if (c == '\r') 
        lits += "\\r";
    else if (c == '\n')
        lits += "\\n";
    else
        lits.append(1, c);

    if (lits.size() % 120 == 0)
        lits.append(1, '\n');
}

void PEncodeRepMatch(uint32_t len, uint32_t idx)
{
    repmatch_idx_cnt[idx]++;
    if (len < 31)
        matchlen_cnt[len]++;
    else
        matchlen_cnt[31]++;
}

void PEncodeMatch(uint32_t len, uint32_t dist)
{
    repmatch_idx_cnt[4]++;
    if (len < 31)
        matchlen_cnt[len]++;
    else
        matchlen_cnt[31]++;
}

void PEncode1BMatch()
{
    match1b_cnt++;
}

void PWriteLog()
{
    printf("Literal %u: \n\n\n\n\n", lits_cnt, lits.c_str());
    for(int i = 0; i < 4; i++)
        printf("Rep Match idx %d cnt: %u\n", i, repmatch_idx_cnt[i]);
    printf("Match cnt: %u\n", repmatch_idx_cnt[4]);
    printf("\n\n1B Match Cnt: %u\n", match1b_cnt);
    for(int i = 0; i < 32; i++) 
        printf("Match len of %d cnt: %u\n", i, matchlen_cnt[i]);
    printf("Match len >= 31 cnt: %u\n", matchlen_cnt[31]);
}


#endif
