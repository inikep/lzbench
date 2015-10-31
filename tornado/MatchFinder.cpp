// LZ77 model
//
// (c) Bulat Ziganshin
// (c) Joachim Henke
// (c) Igor Pavlov
// This code is provided on the GPL license.
// If you need a commercial license to use the code, please write to Bulat.Ziganshin@gmail.com

typedef uint   HashVal;      // Result of hashing function
typedef uint32 PtrVal;       // Pointers to buf stored in HTable
typedef uint32 HintVal;      // Cached bytes from buf stored in HTable

// Maximum number of bytes used for hashing in any match finder.
// If this value will be smaller than real, we can hash bytes in buf that are not yet read
// Also it's number of bytes reserved after bufend in order to simplify p+N<=bufend checks
#define MAX_HASHED_BYTES 12


#ifdef DEBUG
void check_match (BYTE *p, BYTE *q, int len)
{
    if (p==q || memcmp(p,q,len))   printf("Bad match:  ");
}
void print_literal (int pos, BYTE c)
{
    printf (isprint(c)? "%08x: '%c'\n" : "%08x: \\0x%02x\n", pos, c);
}
void print_match (int pos, int len, int dist)
{
    printf ("%08x: %3d %6d\n", pos, len, -dist);
}
#else
#define check_match(p,q,len)
#define print_literal(pos,c)
#define print_match(pos,len,dist)
#endif


// Settings for 4-byte hashing
#define value(p)     value32(p)
// Just for readability
#define MINLEN       min_length()

// Checks that bigDist/64 is bigger than smallDist
static inline bool ChangePair (uint smallDist, uint bigDist)
{
    return (bigDist/64 > smallDist);
}

// Hash function hashing 4..7 bytes
static inline uint hashx (int len, BYTE *p, UINT HashShift, UINT HashMask = ~0)
{
    switch(len)
    {
    case 4:  return ((value32(p)*123456791) >> HashShift) & HashMask;
    case 5:  return ((value32(p)*123456791 + value32(p+1)*789567123) >> HashShift) & HashMask;
    case 6:  return ((value32(p)*123456791 + value32(p+2)*789567123) >> HashShift) & HashMask;
    case 7:  return ((value32(p)*123456791 + value32(p+3)*789567123) >> HashShift) & HashMask;
    default: CHECK (FREEARC_ERRCODE_INTERNAL,  TRUE,  (s,"Fatal error: Unsupported len==%d in the hashx()", len));  return 0;
    }
}

// Check match len/distance for greedy/lazy parsing (distances for small matches need to be limited)
static inline bool accept_len_dist (int len, DISTANCE dist)
{
    switch(len)
    {
    case 4:  return dist <  48*kb;
    case 5:  return dist < 192*kb;
    case 6:  return dist <   1*mb;
    case 7:  return dist <  12*mb;
    default: return true;
    }
}

// Check match at q:len for greedy/lazy parsing (distances for small matches need to be limited)
static inline int accept_match (int len, BYTE *p, BYTE *q, void *bufend)
{
    switch(len)
    {
    case 4:  return p-q<48*kb  && p<=bufend && val32equ(p, q)                                    ? 4 : 0;
    case 5:  return p-q<192*kb && p<=bufend && val32equ(p, q) && p[4]==q[4]                      ? 5 : 0;
    case 6:  return p-q<1*mb   && p<=bufend && val32equ(p, q) && val16equ(p+4, q+4)              ? 6 : 0;
    case 7:  return p-q<12*mb  && p<=bufend && val32equ(p, q) && val24equ(p+4, q+4)              ? 7 : 0;
    case 8:  return               p<=bufend && val32equ(p, q) && val32equ(p+4, q+4)              ? 8 : 0;
    case 9:  return               p<=bufend && val32equ(p, q) && val32equ(p+4, q+4) && p[8]==q[8]? 9 : 0;
    case 10: return               p<=bufend && val32equ(p, q) && val32equ(p+4, q+4) && val16equ(p+8, q+8)? 10 : 0;
    default: CHECK (FREEARC_ERRCODE_INTERNAL,  TRUE,  (s,"Fatal error: Unsupported len==%d in the accept_match()", len));  return 0;
    }
}

// Check match at q:len for optimal parsing
static inline bool accept_optimal_match (int len, BYTE *p, BYTE *q, void *bufend)
{
    switch(len)
    {
    case 4:  return p<=bufend && val32equ(p, q)                                    ;
    case 5:  return p<=bufend && val32equ(p, q) && p[4]==q[4]                      ;
    case 6:  return p<=bufend && val32equ(p, q) && val16equ(p+4, q+4)              ;
    case 7:  return p<=bufend && val32equ(p, q) && val24equ(p+4, q+4)              ;
    case 8:  return p<=bufend && val32equ(p, q) && val32equ(p+4, q+4)              ;
    case 9:  return p<=bufend && val32equ(p, q) && val32equ(p+4, q+4) && p[8]==q[8];
    case 10: return p<=bufend && val32equ(p, q) && val32equ(p+4, q+4) && val16equ(p+8, q+8);
    default: CHECK (FREEARC_ERRCODE_INTERNAL,  TRUE,  (s,"Fatal error: Unsupported len==%d in the accept_optimal_match()", len));  return FALSE;
    }
}

// Returns number of matched bytes up to maxLen
UINT check_match_len (BYTE *p, BYTE *q, UINT maxLen)
{
    UINT len = 0;
    while (val32equ(p+len, q+len))  {len+=4;  if (len>=maxLen) return maxLen;}
    while (p[len]==q[len] && len<maxLen)  len++;
    return len;
}



// ****************************************************************************
// Abstract Match Finder class which defines codebase common for all concrete Match Finders
// ****************************************************************************

struct BaseMatchFinder
{
    UINT HashSize, HashShift, HashMask;
    PtrVal *HTable;        // Hash table usd to quickly find matches
    BYTE *base;            // Base value (offsets to this ptr are stored in HTable)
    BYTE *q;               // Pointer to last match found
    int hash_row_width;

    // Empty contructor
    BaseMatchFinder() {};
    // This contructor is common for several Match Finder classes.
    BaseMatchFinder (BYTE *buf, uint dictsize, uint hashsize, int _hash_row_width, uint auxhash_size, int auxhash_row_width);
    // Returns error code if there is any problem in MF work
    int error();
    // Called on initialization and when next data chunk read into NON-SLIDING buffer
    void clear_hash (BYTE *buf);
    // Called after data was shifted 'shift' bytes backwards in SLIDING WINDOW buf[]
    void shift (BYTE *buf, int shift);
    // Minimal length of matches returned by this Match Finder
    static uint min_length()  {return 4;}
    // Fill the `matches` buffer with len/dist of all found matches
    DISTANCE *find_all_matches (byte *p, void *bufend, DISTANCE *matches)    {return matches;}
    // Returns pointer of last match found (length is returned by find_matchlen() itself)
    byte *get_matchptr()  {return q;}
    // hash function
    uint hash (uint x)    {return ((x*123456791) >> HashShift) & HashMask;}
    // Invalidate previously found match
    void invalidate_match ()   {}
    // Prefetch MF data associated with provided `p`
    void prefetch_hash (BYTE *p)    {}
#ifdef FREEARC_64BIT
    // Convert 32-bit value, stored in HTable, into pointer
    BYTE *toPtr(PtrVal n) {return base+n;}
    // And back
    PtrVal fromPtr(BYTE *p) {return p-base;}
#else
    // In 32-bit world
    BYTE  *toPtr  (PtrVal n) {return (BYTE*) n;}
    PtrVal fromPtr(BYTE  *p) {return (PtrVal) p;}
#endif
};

// This contructor is common for several match finder classes.
// It allocs and inits HTable and inits Hash* fields
BaseMatchFinder::BaseMatchFinder (BYTE *buf, uint dictsize, uint hashsize, int _hash_row_width, uint, int)
{
    base      = buf;
    hash_row_width = _hash_row_width;
    HashSize  = (1<<lb(hashsize)) / sizeof(*HTable);
    HashShift = 32-lb(HashSize);
    HashMask  = (HashSize-1) & ~(roundup_to_power_of(hash_row_width,2)-1);
    HTable    = (PtrVal*) BigAlloc (sizeof(PtrVal) * HashSize);
}

// Returns error code if there is any problem in MF work
int BaseMatchFinder::error()
{
    return HTable==NULL?  FREEARC_ERRCODE_NOT_ENOUGH_MEMORY : FREEARC_OK;
}

// Called when next data chunk read into NON-SLIDING buffer
void BaseMatchFinder::clear_hash (BYTE *buf)
{
    if (HTable)  iterate_var(i,HashSize)  HTable[i] = fromPtr(buf+1);
}

// Called after data was shifted 'shift' bytes backwards in SLIDING WINDOW buf[]
// We should make appropriate corrections in HTable. We set HTable entries minimum
// to buf+1 because with lazy matching we can extend match found one byte backward
void BaseMatchFinder::shift (BYTE *buf, int shift)
{
    iterate_var(i,HashSize)  HTable[i]  =  HTable[i] > fromPtr(buf+shift)?  HTable[i]-shift : fromPtr(buf+1);
}



// ****************************************************************************
// 1-way Match Finder which holds only one entry for each hash value **********
// ****************************************************************************

struct MatchFinder1 : BaseMatchFinder
{
    MatchFinder1 (BYTE *buf, uint dictsize, uint hashsize, int hash_row_width, uint auxhash_size, int auxhash_row_width)
        : BaseMatchFinder (buf, dictsize, hashsize, hash_row_width, auxhash_size, auxhash_row_width)
                     {clear_hash(buf);}
    ~MatchFinder1 () {BigFree(HTable);}

    uint find_matchlen (byte *p, void *bufend, UINT prevlen)
    {
        UINT h = hash(value(p));
        q = toPtr(HTable[h]);  HTable[h] = fromPtr(p);
        if (val32equ(p, q)) {
            UINT len;
            for (len=MINLEN-1; p+len+4<bufend && val32equ(p+len, q+len); len+=4);
            for (            ; p+len  <bufend && p[len] == q[len];       len++);
            return len;
        } else {
            return MINLEN-1;
        }
    }
    void update_hash (BYTE *p, UINT len, UINT step)
    {
// No hash update in fastest mode!
/*        uint h;
        h = hash(value(p+1)),  HTable[h] = p+1;
        p += len;
        h = hash(value(p-2)),  HTable[h] = p-2;
        h = hash(value(p-1)),  HTable[h] = p-1;
*/    }
};



// ****************************************************************************
// 2-way Match Finder *********************************************************
// ****************************************************************************

struct MatchFinder2 : BaseMatchFinder
{
    MatchFinder2 (BYTE *buf, uint dictsize, uint hashsize, int hash_row_width, uint auxhash_size, int auxhash_row_width)
        : BaseMatchFinder (buf, dictsize, hashsize, hash_row_width, auxhash_size, auxhash_row_width)
                     {clear_hash(buf);}
    ~MatchFinder2 () {BigFree(HTable);}

    uint find_matchlen (byte *p, void *bufend, UINT prevlen)
    {
        UINT len;
        UINT h = hash(value(p));
        BYTE *q1 = toPtr (HTable[h+1]);   HTable[h+1] = HTable[h];
              q  = toPtr (HTable[h]);     HTable[h]   = fromPtr(p);
        if (val32equ(p, q)) {
            for (len=MINLEN-1; p+len+4<bufend && val32equ(p+len, q+len); len+=4);
            for (            ; p+len  <bufend && p[len] == q[len];       len++);

            if (p[len] == q1[len]) {
                UINT len1;
                for (len1=0; p+len1<bufend && p[len1]==q1[len1]; len1++);
                if (len1>len)  len=len1, q=q1;
            }
            return len;
        } else if (val32equ(p, q1)) {
            q=q1;
            for (len=MINLEN-1; p+len+4<bufend && val32equ(p+len, q+len); len+=4);
            for (            ; p+len  <bufend && p[len] == q[len];       len++);
            return len;
        } else {
            return MINLEN-1;
        }
    }

    void update_hash (BYTE *p, UINT len, UINT step)
    {   // len may be as low as 1 if LazyMatching and Hash3 are used together
        UINT h;
        h = hash(value(p+1)),  HTable[h+1] = HTable[h],  HTable[h] = fromPtr(p+1);
        p += len;
        h = hash(value(p-2)),  HTable[h+1] = HTable[h],  HTable[h] = fromPtr(p-2);
        h = hash(value(p-1)),  HTable[h+1] = HTable[h],  HTable[h] = fromPtr(p-1);
    }
};



// ****************************************************************************
// Multi-way Match Finder searching for N+ byte matches
// ****************************************************************************

template <uint N>
struct MatchFinderN : BaseMatchFinder
{
    MatchFinderN (BYTE *buf, uint dictsize, uint hashsize, int hash_row_width, uint auxhash_size, int auxhash_row_width)
        : BaseMatchFinder (buf, dictsize, hashsize, hash_row_width, auxhash_size, auxhash_row_width)
                     {clear_hash(buf);}
    ~MatchFinderN () {BigFree(HTable);}

    // Fill the `matches` buffer with len/dist of all found matches
    DISTANCE *find_all_matches (byte *p, void *bufend, DISTANCE *matches)
    {
        UINT h = hashx (N, p, HashShift, HashMask);
        UINT maxLen = MINLEN-1;
        PtrVal x1, x0 = HTable[h];  HTable[h] = fromPtr(p);
        BYTE *q = toPtr(x0);
        // Start with checking the first element of the hash row
        if (val32equ(p, q)) {
            UINT len;
            for (len=4; p+len+4<=bufend && val32equ(p+len, q+len); len+=4);
            for (     ; p+len  < bufend && p[len] == q[len];       len++);
            if (len >= MINLEN) {
                *matches++ = maxLen = len;
                *matches++ = p-q;
            }
        }
        // Check remaining elements, searching for longer match,
        // and shift entire row toward end
        for (int j=1; j<hash_row_width; j++, x0=x1) {
            x1=HTable[h+j];  HTable[h+j]=x0;
            BYTE *q = toPtr(x1);
            if (p[maxLen] == q[maxLen]) {
                UINT len;
                for (len=0;  p+len<bufend && p[len]==q[len];  len++);
                if (len >= maxLen) {
                    *matches++ = maxLen = len;
                    *matches++ = p-q;
                }
            }
        }
        return matches;
    }

    uint find_matchlen (byte *p, void *bufend, UINT prevlen)
    {
        UINT h = hashx (N, p, HashShift, HashMask);
        UINT len = MINLEN-1;
        PtrVal x1, x0 = HTable[h];  HTable[h] = fromPtr(p);
        q = toPtr(x0);
        // Start with checking the first element of the hash row
        if (val32equ(p, q)) {
            for (len=MINLEN; p+len+4<=bufend && val32equ(p+len, q+len); len+=4);
            for (          ; p+len  < bufend && p[len] == q[len];       len++);
            if (!accept_len_dist(len,p-q))
              len = MINLEN-1;
        }
        // Check remaining elements, searching for longer match,
        // and shift entire row toward end
        for (int j=1; j<hash_row_width; j++, x0=x1) {
            x1=HTable[h+j];  HTable[h+j]=x0;
            BYTE *q1 = toPtr(x1);
            if (val32equ(p+len+1-MINLEN, q1+len+1-MINLEN)) {
                UINT len1;
                for (len1=0;  p+len1<bufend && p[len1]==q1[len1];  len1++);
                if (len1 > len  &&  !(len1==len+1 && ChangePair(p-q, p-q1))  &&  accept_len_dist(len1,p-q1))
                    len=len1, q=q1;
            }
        }
        return len;
    }

    void update_hash1 (BYTE *p)
    {
        UINT h = hashx (N, p, HashShift, HashMask);
        for (int j=hash_row_width; --j; )
            HTable[h+j] = HTable[h+j-1];
        HTable[h] = fromPtr(p);
    }
    void update_hash (BYTE *p, UINT len, UINT step)
    {
        if (len>1)                         update_hash1 (p+1);
        for (int i=2; i<len-1; i+=step)    update_hash1 (p+i);
        if (len>3)                         update_hash1 (p+len-1);
    }
    // Minimal length of matches returned by this Match Finder
    static uint min_length()  {return N;}
};



// ****************************************************************************
// Multi-way Match Finder searching for N-byte matches.
// Only first hash element is replaced in update_hash1()
// ****************************************************************************

template <uint N>
struct ExactMatchFinder : BaseMatchFinder
{
    ExactMatchFinder (BYTE *buf, uint dictsize, uint hashsize, int hash_row_width, uint auxhash_size, int auxhash_row_width)
        : BaseMatchFinder (buf, dictsize, hashsize, hash_row_width, auxhash_size, auxhash_row_width)
                         {clear_hash(buf);}
    ~ExactMatchFinder () {BigFree(HTable);}

    // Fill the `matches` buffer with len/dist of all found matches
    DISTANCE *find_all_matches (byte *p, void *bufend, DISTANCE *matches)
    {
        UINT h = hashx (N, p, HashShift, HashMask);
        PtrVal x1, x0 = fromPtr(p);
        // Check hash row elements, searching for N-byte match,
        // and shift entire row toward end
        for (int j=0; j<hash_row_width; j++, x0=x1) {
            x1=HTable[h+j];  HTable[h+j]=x0;
            BYTE *q = toPtr(x1);
            if (accept_optimal_match (N, p, q, bufend)) {
                *matches++ = N;
                *matches++ = p-q;
                return matches;
            }
        }
        return matches;
    }

    uint find_matchlen (byte *p, void *bufend, UINT prevlen)
    {
        UINT h = hashx (N, p, HashShift, HashMask);
        PtrVal x1, x0 = fromPtr(p);
        // Check hash row elements, searching for N-byte match,
        // and shift entire row toward end
        for (int j=0; j<hash_row_width; j++, x0=x1) {
            x1=HTable[h+j];  HTable[h+j]=x0;
            BYTE *q1 = toPtr(x1);
            if (!accept_len_dist (N, p-q1))
                return MINLEN-1;
            if (accept_optimal_match (N, p, q1, bufend)) {
                q=q1; return N;
            }
        }
        return MINLEN-1;
    }

    void update_hash1 (BYTE *p)
    {
        UINT h = hashx (N, p, HashShift, HashMask);
        HTable[h] = fromPtr(p);
    }
    void update_hash (BYTE *p, UINT len, UINT step)
    {
        if (len>1)                         update_hash1 (p+1);
        for (int i=2; i<len-1; i+=step)    update_hash1 (p+i);
        if (len>3)                         update_hash1 (p+len-1);
    }
    // Minimal length of matches returned by this Match Finder
    static uint min_length()  {return N;}
};



// ****************************************************************************
// This matchfinder caches 4 bytes of string (p[3..6]) in hash itself,
// providing faster search in case of highly-populated hash rows.
// For efficiency reasons I suggest to use it only for hash_row_width>=4,
// buffer>=8mb and hashsize>=2mb.
// This class is compatible only with MINLEN>=4
// There are two possible implementation strategies:
//    a) guarantee that caching hash entries contains valid data and use this fact to cut off some tests
//    b) don't make such guarantees and make strict tests of actual data in buf[]
// I have not yet decided which strategy to use :)
// ****************************************************************************

template <uint N>  // N is minimum match length (4..7)
struct CachingMatchFinder : BaseMatchFinder
{
    int hash_row_width2, hash_row_width_up;

    CachingMatchFinder (BYTE *buf, uint dictsize, uint hashsize, int _hash_row_width, uint auxhash_size, int auxhash_row_width);
    ~CachingMatchFinder() {BigFree(HTable);}

    void clear_hash (BYTE *buf);
    void shift (BYTE *buf, int shift);
    uint key (BYTE *p)  {return value32(p+N-1);}   // Key value stored in hash for each string

    // Prefetch MF data associated with provided `p`
    void prefetch_hash (BYTE *p)
    {
        // Hash key of string for which we will perform match search
        UINT h = hashx(N,p,HashShift);
        // Pointers to the current hash element and end of hash row
        PtrVal *table = HTable+h*hash_row_width2,  *tabend = table + hash_row_width2;
        for (;  table < tabend;  table += CACHE_ROW/sizeof(*table))
            prefetch (*table);
    }

// Read next ptr/key from hash and save here previous pair (shifted from previous position)
#define next_pair()                                       \
            PtrVal x0=x1;  x1 = *table;  *table++ = x0;   \
            UINT   v0=v1;  v1 = *table;  *table++ = v0;   \
            UINT t = v1 ^ key(p);

#define SAVE_MATCH(len,dist)  (*matches++ = (len),  *matches++ = (dist))

    // Fill the `matches` buffer with len/dist of all found matches
    DISTANCE *find_all_matches (byte *p, void *bufend, DISTANCE *matches)
    {
        UINT len;                         // Length of largest string already found
        UINT h = hashx(N,p,HashShift);    // Hash key of the string for which we perform the match search
        // Pointers to the current hash element and end of hash row
        PtrVal *table = HTable+h*hash_row_width2, *tabend = table + hash_row_width2;
        // x1/v1 keeps last match candidate and its key (we init them with values which would be saved in the first hash slot)
        PtrVal x1 = fromPtr(p);
        UINT   v1 = key(p);
        BYTE  *q1;

        // Check hash elements, searching for longest match,
        // while shifting entire row contents towards end
        // (there are five loops here - one used before any match is found,
        //  three are used when a match of size 4/5/6 already found,
        //  and one used when match of size 7+ already found)
//len0:
        while (table!=tabend) {
            next_pair();
            if ((t&0xff) == 0) {
                q1 = toPtr(x1);
                     if (t==0        && accept_optimal_match(N+3,p,q1,bufend))    goto len7;
                else if (t&0xff00    && accept_optimal_match(N,  p,q1,bufend))    {SAVE_MATCH(N,  p-q1);  goto len4;}
                else if (t&0xff0000  && accept_optimal_match(N+1,p,q1,bufend))    {SAVE_MATCH(N+1,p-q1);  goto len5;}
                else if (               accept_optimal_match(N+2,p,q1,bufend))    {SAVE_MATCH(N+2,p-q1);  goto len6;}
            }
        }
        return matches;

len4:   while (table!=tabend) {
            next_pair();
            if ((t&0xffff) == 0) {
                q1 = toPtr(x1);
                     if (t==0        && accept_optimal_match(N+3,p,q1,bufend))    goto len7;
                else if (t&0xff0000  && accept_optimal_match(N+1,p,q1,bufend))    {SAVE_MATCH(N+1,p-q1);  goto len5;}
                else if (               accept_optimal_match(N+2,p,q1,bufend))    {SAVE_MATCH(N+2,p-q1);  goto len6;}
            }
        }
        return matches;

len5:   while (table!=tabend) {
            next_pair();
            if ((t&0xffffff) == 0) {
                q1 = toPtr(x1);
                     if (t==0        && accept_optimal_match(N+3,p,q1,bufend))    goto len7;
                else if (               accept_optimal_match(N+2,p,q1,bufend))    {SAVE_MATCH(N+2,p-q1);  goto len6;}
            }
        }
        return matches;

len6:   while (table!=tabend) {
            next_pair();
            q1 = toPtr(x1);
            if (t==0 && accept_optimal_match(N+3,p,q1,bufend))    goto len7;
        }
        return matches;

len7:   len = N+3;
        for (; p+len<bufend && val32equ(p+len, q1+len); len+=4);
        for (; p+len<bufend && p[len] == q1[len];       len++);
        SAVE_MATCH(len,p-q1);

        while (table!=tabend) {
            next_pair();
            BYTE *q1 = toPtr(x1);
            if (t == 0 && p[len] == q1[len] && val32equ(p, q1)) {
                UINT len1 = 4;
                for (; p+len1<bufend && val32equ(p+len1, q1+len1); len1+=4);
                for (; p+len1<bufend && p[len1] == q1[len1];       len1++);
                if (len1>len)  SAVE_MATCH(len=len1,p-q1);
            }
        }
        return matches;
    }

    uint find_matchlen (byte *p, void *bufend, UINT prevlen)
    {
        UINT len;                         // Length of largest string already found
        UINT h = hashx(N,p,HashShift);    // Hash key of string for which we perform match search
        // Pointers to the current hash element and end of hash row
        PtrVal *table = HTable+h*hash_row_width2, *tabend = table + hash_row_width2;
        // x1/v1 keeps last match candidate and its key (we init them with values which would be saved in the first hash slot)
        PtrVal x1 = fromPtr(p);
        UINT   v1 = key(p);

        // Check hash elements, searching for longest match,
        // while shifting entire row contents towards end
        // (there are five loops here - one used before any match is found,
        //  three are used when a match of size 4/5/6 already found,
        //  and one used when match of size 7+ already found)
//len0:
        while (table!=tabend) {
            next_pair();
            if ((t&0xff) == 0) {
                     if (t==0)        goto len7;
                else if (t&0xff00)    goto len4;
                else if (t&0xff0000)  goto len5;
                else                  goto len6;
            }
        }
        return MINLEN-1;

len4:   q = toPtr(x1);
        while (table!=tabend) {
            next_pair();
            if ((t&0xffff) == 0) {
                     if (t==0)        goto len7;
                else if (t&0xff0000)  goto len5;
                else                  goto len6;
            }
        }
        return accept_match(N, p, q, bufend);

len5:   q = toPtr(x1);
        while (table!=tabend) {
            next_pair();
            if ((t&0xffffff) == 0) {
                     if (t==0)        goto len7;
                else                  goto len6;
            }
        }
        return accept_match(N+1, p, q, bufend);

len6:   q = toPtr(x1);
        while (table!=tabend) {
            next_pair();
            if (t == 0)                goto len7;
        }
        return accept_match(N+2, p, q, bufend);

len7:   q = toPtr(x1);
        len = MINLEN-1;
        if (val32equ(p, q)) {
            len = mymin(MINLEN-1, 4);
            for (; p+len<bufend && val32equ(p+len, q+len); len+=4);
            for (; p+len<bufend && p[len] == q[len];       len++);
        }

        while (table!=tabend) {
            next_pair();
            BYTE *q1 = toPtr(x1);
            if (t == 0 && p[len] == q1[len] && val32equ(p, q1)) {
                UINT len1 = mymin(MINLEN-1, 4);
                for (; p+len1<bufend && val32equ(p+len1, q1+len1); len1+=4);
                for (; p+len1<bufend && p[len1] == q1[len1];       len1++);
                if (len1>len)  len=len1, q=q1;
            }
        }
        return len;
    }

#undef next_pair
#undef SAVE_MATCH

    // Update half of hash row corresponding to string pointed by p
    // (hash updated via this procedure only when skipping match contents)
    void update_hash1 (BYTE *p)
    {
        UINT h = hashx(N,p,HashShift)*hash_row_width2;    // Hash key of string at p
        for (int j=hash_row_width_up; j-=2; )
            HTable[h+j]   = HTable[h+j-2],
            HTable[h+j+1] = HTable[h+j-1];
        HTable[h]   = fromPtr(p);
        HTable[h+1] = key(p);
    }
    // Skip match starting from p with length len and update hash with strings using given step
    void update_hash (BYTE *p, UINT len, UINT step)
    {
        // Update hash with strings at the start and end of match plus part of strings inside match
        if (len>1)                         update_hash1 (p+1);
        for (int i=2; i<len-1; i+=step)    update_hash1 (p+i);
        if (len>3)                         update_hash1 (p+len-1);
    }
    // Minimal length of matches returned by this Match Finder
    static uint min_length()  {return N;}
};

template <uint N>
CachingMatchFinder<N>::CachingMatchFinder (BYTE *buf, uint dictsize, uint hashsize, int _hash_row_width, uint auxhash_size, int auxhash_row_width)
{
    base      = buf;
    hash_row_width = _hash_row_width;
    // Simulate 2gb:256 hash with 2040mb:255 one :)
    if (hashsize==2*gb  &&  hash_row_width == 1<<lb(hash_row_width))
        --hash_row_width;
    hash_row_width2 = hash_row_width*2;
    hash_row_width_up = hash_row_width + (hash_row_width % 2);  // increment odd values by 1

    UINT rows = 1 << lb(hashsize / (sizeof(*HTable) * hash_row_width2));
    HashSize  = rows * hash_row_width2;
    HTable    = (PtrVal*) BigAlloc (HashSize * sizeof(*HTable));

    HashShift = 32 - lb(rows);
    HashMask  = ~0;
    if (error() == FREEARC_OK)
        clear_hash (buf);
}

template <uint N>
void CachingMatchFinder<N>::clear_hash (BYTE *buf)
{
    for (int i=0; i<HashSize; i+=2)
        HTable[i]   = fromPtr(buf+1),
        HTable[i+1] = key(buf+1);
}

template <uint N>
void CachingMatchFinder<N>::shift (BYTE *buf, int shift)
{
    for (int i=0; i<HashSize; i+=2)
        HTable[i]  =  HTable[i] > fromPtr(buf+shift)?  HTable[i]-shift  :  fromPtr(buf+1);
}



// *****************************************************************************************************
// One more caching match finder. This one doesn't shift data in hash rows, but keeps head index instead
// *****************************************************************************************************

template <uint N>  // N is minimum match length (4..7)
struct CycledCachingMatchFinder : BaseMatchFinder
{
    BYTE* Head;
    UINT  HeadSize;

    CycledCachingMatchFinder (BYTE *buf, uint dictsize, uint hashsize, int _hash_row_width, uint auxhash_size, int auxhash_row_width);
    ~CycledCachingMatchFinder()  {BigFree(HTable);  BigFree(Head);}
    int error()  {return HTable==NULL || Head==NULL?  FREEARC_ERRCODE_NOT_ENOUGH_MEMORY : FREEARC_OK;}    // Returns error code if there is any problem in MF work

    void clear_hash (BYTE *buf);
    void shift (BYTE *buf, int shift);
    uint key (BYTE *p)  {return value32(p+N-1);}   // Key value stored in hash for each string

    // Fill the `matches` buffer with len/dist of all found matches
    DISTANCE *find_all_matches (byte *p, void *bufend, DISTANCE *matches)
    {
        UINT len = find_matchlen (p, bufend, 0);
               q = get_matchptr();
        if (len >= MINLEN) {
            *matches++ = len;
            *matches++ = p-q;
        }
        return matches;
    }

    uint find_matchlen (byte *p, void *bufend, UINT prevlen)
    {
        UINT len;                         // Length of largest string already found
        UINT h = hashx(N,p,HashShift);    // Hash key of string for which we perform match search
        UINT i = (Head[h]  =  Head[h]==0?  hash_row_width-1  :  Head[h]-1);  // index of new head of this hash row
        PtrVal x1;
        // Pointers to the current hash element and end of hash row
        PtrVal *rowstart = HTable+h*hash_row_width*2,  *rowend = rowstart + hash_row_width*2;
        PtrVal *table = rowstart + i*2,  *tabend = table;
        // Save current ptr and its key to the first hash slot
        *table++ = fromPtr(p);
        *table++ = key(p);
        if (table==rowend)  table = rowstart;

        // Check hash elements, searching for longest match
        // (there are five loops here - one used before any match is found,
        //  three are used when a match of size N/N+1/N+2 already found,
        //  and last one used when match of size >=N+3 already found)
// Read next ptr/key pair from hash
#define next_pair()                                                    \
            x1 = *table++;  if(!x1) break;                             \
            UINT v1 = *table++;                                        \
            if (table==rowend)  table = rowstart;                      \
            UINT t = v1 ^ key(p);

//len0:
        while (table!=tabend) {
            next_pair();
            if ((t&0xff) == 0) {
                     if (t==0)        goto len7;
                else if (t&0xff00)    goto len4;
                else if (t&0xff0000)  goto len5;
                else                  goto len6;
            }
        }
        return MINLEN-1;

len4:   q = toPtr(x1);
        while (table!=tabend) {
            next_pair();
            if ((t&0xffff) == 0) {
                     if (t==0)        goto len7;
                else if (t&0xff0000)  {if (!ChangePair(p-q, p-toPtr(x1)))  goto len5;}
                else                  goto len6;
            }
        }
        return accept_match(N, p, q, bufend);

len5:   q = toPtr(x1);
        while (table!=tabend) {
            next_pair();
            if ((t&0xffffff) == 0) {
                     if (t==0)        goto len7;
                else                  {if (!ChangePair(p-q, p-toPtr(x1)))  goto len6;}
            }
        }
        return accept_match(N+1, p, q, bufend);

len6:   q = toPtr(x1);
        while (table!=tabend) {
            next_pair();
            if (t==0) {
                BYTE *q1 = toPtr(x1);
                if (val32equ(p+N, q1+N) || !ChangePair(p-q, p-q1))  goto len7;
            }
        }
        return accept_match(N+2, p, q, bufend);

len7:   len = MINLEN-1;
        q = toPtr(x1);
        if (val32equ(p, q)) {
            len = mymin(MINLEN-1, 4);
            for (; p+len<bufend && val32equ(p+len, q+len); len+=4);
            for (; p+len<bufend && p[len] == q[len];       len++);
        }

        while (table!=tabend) {
            next_pair();
            BYTE *q1 = toPtr(x1);
            if (t == 0 && p[len] == q1[len] && val32equ(p, q1)) {
                UINT len1 = mymin(MINLEN-1, 4);
                for (; p+len1<bufend && val32equ(p+len1, q1+len1); len1+=4);
                for (; p+len1<bufend && p[len1] == q1[len1];       len1++);
                if (len1>len  &&  !(len1==len+1 && ChangePair(p-q, p-q1)) )
                    len=len1, q=q1;
            }
        }
        return len;
#undef next_pair
    }

    // Update hash row corresponding to string pointed by p
    // (hash updated via this procedure only when skipping match contents)
    void update_hash1 (BYTE *p)
    {
        UINT h = hashx(N,p,HashShift);    // Hash key of string at p
        UINT i = (Head[h] = Head[h]==0? hash_row_width-1 : Head[h]-1);  // index of new head of this hash row
        PtrVal *table = HTable + (h*hash_row_width+i)*2;
        table[0] = fromPtr(p);
        table[1] = key(p);
    }
    // Skip match starting from p with length len and update hash with strings using given step
    void update_hash (BYTE *p, UINT len, UINT step)
    {
        for (int i=1; i<len; i++)
            update_hash1 (p+i);
    }
    // Minimal length of matches returned by this Match Finder
    static uint min_length()  {return N;}
};

template <uint N>
CycledCachingMatchFinder<N>::CycledCachingMatchFinder (BYTE *buf, uint dictsize, uint hashsize, int _hash_row_width, uint auxhash_size, int auxhash_row_width)
{
    base      = buf;
    hash_row_width = _hash_row_width;
    // Simulate 2gb:256 hash with 2040mb:255 one :)
    if (hashsize==2*gb  &&  hash_row_width == 1<<lb(hash_row_width))
        --hash_row_width;

    HeadSize  = 1 << lb(hashsize / (sizeof(*HTable) * hash_row_width * 2));
    Head      = (BYTE*)  BigAlloc (HeadSize * sizeof(*Head));

    HashSize  = HeadSize * hash_row_width * 2;
    HTable    = (PtrVal*) BigAlloc (HashSize * sizeof(*HTable));

    HashShift = 32 - lb(HeadSize);
    HashMask  = ~0;
    if (error() == FREEARC_OK)
        clear_hash (buf);
}

template <uint N>
void CycledCachingMatchFinder<N>::clear_hash (BYTE *buf)
{
    iterate_var(i,HashSize)  HTable[i]=0;
    iterate_var(i,HeadSize)  Head  [i]=0;
}

template <uint N>
void CycledCachingMatchFinder<N>::shift (BYTE *buf, int shift)
{
    for (int i=0; i<HashSize; i+=2)
        HTable[i]  =  HTable[i] > fromPtr(buf+shift)?  HTable[i]-shift  :  0;
}



// ****************************************************************************
// Binary Tree match finder from LZMA/LzFind.c (c) Igor Pavlov ****************
// ****************************************************************************

template <uint N>
struct BinaryTreeMatchFinder : BaseMatchFinder
{
    static const UINT kEmptyHashValue = 0;
    UINT  DictSize;     // Size of the match search window
    UINT *BinaryTree;   // Binary trees of matches having the same hash value
    int   MaxDepth;     // Maximum depth of search in the binary tree
    BYTE *LastBufEnd;   // Last `bufend` value passed to find_matchlen() used by update_hash()

    BinaryTreeMatchFinder (BYTE *buf, uint dictsize, uint hashsize, int maxdepth, uint auxhash_size, int auxhash_row_width)
        : BaseMatchFinder (buf, dictsize, hashsize, 1, auxhash_size, auxhash_row_width),  DictSize(dictsize),  MaxDepth(maxdepth)
    {
        BinaryTree = (UINT*) BigAlloc (2 * sizeof(UINT) * DictSize);  // for every dictionary position, we save pointers to the left and right "sons" as UINT indexes
        if (error() == FREEARC_OK)
            clear_hash(buf);
    }
    ~BinaryTreeMatchFinder()
    {
        BigFree(BinaryTree);
        BigFree(HTable);
    }
    // Returns error code if there is any problem in MF work
    int error()  {return BinaryTree==NULL?  FREEARC_ERRCODE_NOT_ENOUGH_MEMORY : BaseMatchFinder::error();}

    // Called on initialization and when next data chunk read into NON-SLIDING buffer
    void clear_hash (BYTE *buf);
    // Called after data was shifted 'shift' bytes backward in SLIDING WINDOW buf[]
    void shift (BYTE *buf, int shift);
    // Minimal length of matches returned by this Match Finder
    static uint min_length()  {return N;}

    // Fill the `matches` buffer with len/dist of all found matches
    DISTANCE *find_all_matches (byte *p, void *bufend, DISTANCE *matches)
    {
        int  depth  = MaxDepth;                // Limit to amount of matches checked
        UINT maxLen = MINLEN-1;                // Length of largest string found so far
        UINT h      = hashx(N,p,HashShift);    // Hash key of string for which we perform match search
        UINT n      = UINT(HTable[h]);         // Index of the current match checked (relative to `base`)
        HTable[h]   = PtrVal(p-base);
        UINT *ptr1  = & BinaryTree[size_t(p-base)*2],  *ptr0 = ptr1 + 1;
        UINT len0 = 0,  len1 = 0,  lenLimit = (byte*)bufend-p;
        for (;;)
        {
          if (depth-- == 0  ||  n == kEmptyHashValue)    { *ptr0 = *ptr1 = kEmptyHashValue;  break; }
          UINT *pair = & BinaryTree[size_t(n)*2];  prefetch(*pair);
          BYTE *q    = base+n;
          UINT  len  = mymin (len0, len1);
          if (q[len] == p[len])
          {
            if (++len != lenLimit && q[len] == p[len])
              while (++len != lenLimit)
                if (q[len] != p[len])
                  break;
            if (len > maxLen)
            {
              *matches++ = maxLen = len;
              *matches++ = p-q;
              if (len == lenLimit) { *ptr1 = pair[0];  *ptr0 = pair[1];  break; }
            }
          }
          if (q[len] < p[len])    *ptr1 = n,  ptr1 = pair + 1,  n = *ptr1,  len1 = len;
          else                    *ptr0 = n,  ptr0 = pair,      n = *ptr0,  len0 = len;
        }
        return matches;
    }

    uint find_matchlen (byte *p, void *bufend, UINT prevlen)
    {
        int  depth  = MaxDepth;                // Limit to amount of matches checked
        UINT maxLen = MINLEN-1;                // Length of largest string found so far
        UINT h      = hashx(N,p,HashShift);    // Hash key of string for which we perform match search
        UINT n      = UINT(HTable[h]);         // Index of the current match checked (relative to `base`)
        HTable[h]   = PtrVal(p-base);
        UINT *ptr1  = & BinaryTree[size_t(p-base)*2],  *ptr0 = ptr1 + 1;
        UINT len0 = 0,  len1 = 0,  lenLimit = (byte*)bufend-p;  LastBufEnd = (byte*)bufend;
        if (bufend <= p  ||  lenLimit <= maxLen)  { *ptr0 = *ptr1 = kEmptyHashValue;  return maxLen; }
        for (;;)
        {
          if (depth-- == 0  ||  n == kEmptyHashValue)    { *ptr0 = *ptr1 = kEmptyHashValue;  break; }
          UINT *pair = & BinaryTree[size_t(n)*2];  prefetch(*pair);
          BYTE *q1   = base+n;
          UINT  len  = mymin (len0, len1);
          if (q1[len] == p[len])
          {
            if (++len != lenLimit && q1[len] == p[len])
              while (++len != lenLimit)
                if (q1[len] != p[len])
                  break;
            if (len > maxLen)
            {
              if (accept_len_dist(len,p-q1)  &&  !(len==maxLen+1 && ChangePair(p-q, p-q1))) {
                maxLen = len;
                q = q1;
              }
              if (len == lenLimit) { *ptr1 = pair[0];  *ptr0 = pair[1];  break; }
            }
          }
          if (q1[len] < p[len])    *ptr1 = n,  ptr1 = pair + 1,  n = *ptr1,  len1 = len;
          else                     *ptr0 = n,  ptr0 = pair,      n = *ptr0,  len0 = len;
        }
        return maxLen;
    }

    // Update hash row corresponding to string pointed by p
    // (hash updated via this procedure only when skipping match contents)
    void update_hash1 (BYTE *p, UINT maxLen = 256*kb)
    {
        void *bufend = (LastBufEnd > p  &&  LastBufEnd-p > maxLen)?  p + maxLen  :  LastBufEnd;
        find_matchlen (p, bufend, MINLEN-1);
    }
    // Skip match starting from p with length len and update hash with strings using given step
    void update_hash (BYTE *p, UINT len, UINT step)
    {
        step  =  mymax (step, len/2503);   // optimization to avoid O(n^2) time behavior after a very long match
        if (len>1)                         update_hash1 (p+1);
        for (int i=2; i<len-1; i+=step)    update_hash1 (p+i);
        if (len>3)                         update_hash1 (p+len-1);
    }
};


// Called when next data chunk read into NON-SLIDING buffer
template <uint N>
void BinaryTreeMatchFinder<N>::clear_hash (BYTE *buf)
{
    if (HTable)  iterate_var(i,HashSize)  HTable[i] = PtrVal(kEmptyHashValue);
    // No need to initialize the BinaryTree[]
}

// Called after data was shifted 'shift' bytes backwards in SLIDING WINDOW buf[].
// We make appropriate corrections in HTable and BinaryTree, reducing and moving
// values and replacing pointers to data shifted out with kEmptyHashValue.
template <uint N>
void BinaryTreeMatchFinder<N>::shift (BYTE *buf, int shift)
{
    iterate_var(i,HashSize)                             HTable    [i]         = PtrVal( UINT(HTable[i]) > shift?  UINT(HTable[i])-shift  :  kEmptyHashValue);
    for (size_t i=shift*2; i<size_t(DictSize)*2; i++)   BinaryTree[i-shift*2] =          BinaryTree[i]  > shift?   BinaryTree[i] -shift  :  kEmptyHashValue;
}



// ****************************************************************************
// This is MF transformer which makes lazy MF from ordinary one ***************
// ****************************************************************************

template <class MatchFinder>
struct LazyMatching
{
    MatchFinder mf;
    uint nextlen;
    BYTE *prevq, *nextq;

    LazyMatching (BYTE *buf, uint dictsize, uint hashsize, int hash_row_width, uint auxhash_size, int auxhash_row_width)
        : mf (buf, dictsize, hashsize, hash_row_width, auxhash_size, auxhash_row_width)
    {
        nextlen = 0;
    }
    void clear_hash (BYTE *buf)
    {
        mf.clear_hash (buf);
        nextlen = 0;
    }

    void shift (BYTE *buf, int shift)
    {
        mf.shift (buf, shift);
        nextq -= shift;
    }

    static uint min_length()  {return MatchFinder::min_length();}
    int  error()          {return mf.error();}
    byte *get_matchptr()  {return prevq;}

    uint find_matchlen (byte *p, void *bufend, UINT _prevlen)
    {
        // Find first match at new position
        if (!nextlen) {
            nextlen = mf.find_matchlen (p, bufend, _prevlen);
            nextq   = mf.get_matchptr();
        }
        // Copy "next match" into current one
        uint prevlen = nextlen;
             prevq   = nextq;
        // Find match at next position
        nextlen = mf.find_matchlen (p+1, bufend, prevlen);
        nextq   = mf.get_matchptr();

        uint nextdist = p+1-nextq;
        uint prevdist = p-prevq;

        // Extend match at p+1 one char backward if it's better than match at p
        if (nextlen>=MINLEN && nextq[-1]==*p &&
               (nextlen+1 >= prevlen && nextdist < prevdist ||
                nextlen+1 == prevlen + 1 && !ChangePair(prevdist, nextdist) ||
                nextlen+1 > prevlen + 1 ||
                nextlen+1 + 1 >= prevlen && prevlen >= MINLEN && ChangePair(nextdist, prevdist)))
        {
            debug (printf ("Extending %d:%d -> %d:%d\n",  nextlen, nextq-p, nextlen+1, nextq-1-p));
            prevlen = nextlen+1;
            prevq   = nextq-1;
            return prevlen;
        }

        // Truncate current match if match at next position will be better (LZMA's algorithm)
        if (nextlen >= prevlen && nextdist < prevdist/4 ||
            nextlen == prevlen + 1 && !ChangePair(prevdist, nextdist) ||
            nextlen > prevlen + 1 ||
            nextlen + 1 >= prevlen && prevlen >= MINLEN && ChangePair(nextdist, prevdist))
        {
             return MINLEN-1;
        } else {
             return prevlen;
        }
    }

    void update_hash (BYTE *p, UINT len, UINT step)
    {
        mf.update_hash (p+1, len-1, step);
        nextlen = 0;
    }

    // Invalidate previously found match
    void invalidate_match ()
    {
        mf.invalidate_match();
        nextlen = MINLEN-1;     // We can't assign 0 because pointer to p already inserted into hash - find_match may return q==p
    }
};



// ****************************************************************************
// This is MF transformer which adds separate small hashes for searching 2/3-byte strings
// ****************************************************************************

template <class MatchFinder, int HASH3_LOG, int HASH2_LOG, bool FULL_UPDATE>
struct Hash3
{
    MatchFinder mf;
    UINT HashSize, HashSize2;
    BYTE **HTable, **HTable2, *q;

    Hash3 (BYTE *buf, uint dictsize, uint hashsize, int hash_row_width, uint auxhash_size, int auxhash_row_width);
    ~Hash3();
    int  error();
    void clear_hash  (BYTE *buf);         // Clear all hash tables
    void clear_hash3 (BYTE *buf);         // Clear only its own hash tables
    void shift (BYTE *buf, int shift);

    static uint min_length()  {return 2;}
    byte *get_matchptr()  {return q;}
    uint hash (uint x)    {return (x*234567913) >> (32-HASH3_LOG);}
    uint hash2(uint x)    {return (x*123456791) >> (32-HASH2_LOG);}

    // Prefetch MF data associated with provided `p` (we suppose that Hash3 own data are small and therefore are already cached)
    void prefetch_hash (BYTE *p)    {mf.prefetch_hash(p);}

    // Fill the `matches` buffer with len/dist of all found matches
    DISTANCE *find_all_matches (byte *p, void *bufend, DISTANCE *matches)
    {
        UINT h = hash2(value16(p));
        BYTE *q = HTable2[h];  HTable2[h] = p;
        if (p+2<=bufend && val16equ(p, q)) {
            *matches++ = 2;
            *matches++ = p-q;
        }
        h = hash(value24(p));
        q = HTable[h];  HTable[h] = p;
        if (p+3<=bufend && val24equ(p, q)) {
            *matches++ = 3;
            *matches++ = p-q;
        }
        return mf.find_all_matches (p, bufend, matches);
    }

    uint find_matchlen (byte *p, void *bufend, UINT prevlen)
    {
        // Find long match
        UINT len = mf.find_matchlen (p, bufend, prevlen);
               q = mf.get_matchptr();
        // If long match was not found - try to find 2/3-byte one
        if (/*prevlen<MINLEN && */  len < mf.MINLEN) {
            UINT h = hash(value24(p));
            q = HTable[h];  HTable[h] = p;
            if (p-q<6*kb && p+3<=bufend && val24equ(p, q)) {
                UINT h = hash2(value16(p));  HTable2[h] = p;
                return 3;
            } else {
                UINT h = hash2(value16(p));
                q = HTable2[h];  HTable2[h] = p;
                if (p-q<256 && p+2<=bufend && val16equ(p, q)) {
                    return 2;
                } else {
                    return MINLEN-1;
                }
            }
        }
        // Update 3-byte & 2-byte hashes
        UINT h = hash(value24(p));  HTable[h] = p;
             h = hash2(value16(p)); HTable2[h] = p;
        // And return 4+-byte match
        return len;
    }

    void update_hash1 (BYTE *p)
    {
        UINT              h = hash(value24(p));  HTable[h]  = p;
        if (FULL_UPDATE) {h = hash2(value16(p)); HTable2[h] = p;}
    }
    void update_hash (BYTE *p, UINT len, UINT step)
    {
        mf.update_hash (p, len, step);
        if (FULL_UPDATE) {
            for (int i=1; i<len; i++)
                update_hash1 (p+i);
        } else {
            if (len>1)  update_hash1 (p+1);
            if (len>3)  update_hash1 (p+len-1);
        }
    }
    // Invalidate previously found match
    void invalidate_match ()   {mf.invalidate_match();}
};

template <class MatchFinder, int HASH3_LOG, int HASH2_LOG, bool FULL_UPDATE>
Hash3<MatchFinder, HASH3_LOG, HASH2_LOG, FULL_UPDATE>
  :: Hash3 (BYTE *buf, uint dictsize, uint hashsize, int hash_row_width, uint auxhash_size, int auxhash_row_width)
   :  mf (buf, dictsize, hashsize, hash_row_width, auxhash_size, auxhash_row_width)
{
    HashSize  = 1 << HASH3_LOG;
    HashSize2 = 1 << HASH2_LOG;
    HTable  = (BYTE**) MidAlloc (sizeof(BYTE*) * HashSize);
    HTable2 = (BYTE**) MidAlloc (sizeof(BYTE*) * HashSize2);
    clear_hash3 (buf);
}

template <class MatchFinder, int HASH3_LOG, int HASH2_LOG, bool FULL_UPDATE>
Hash3<MatchFinder, HASH3_LOG, HASH2_LOG, FULL_UPDATE> :: ~Hash3()
{
    MidFree(HTable);
    MidFree(HTable2);
}

template <class MatchFinder, int HASH3_LOG, int HASH2_LOG, bool FULL_UPDATE>
int Hash3<MatchFinder, HASH3_LOG, HASH2_LOG, FULL_UPDATE> :: error()
{
    return HTable==NULL || HTable2==NULL?  FREEARC_ERRCODE_NOT_ENOUGH_MEMORY : mf.error();
}

template <class MatchFinder, int HASH3_LOG, int HASH2_LOG, bool FULL_UPDATE>
void Hash3<MatchFinder, HASH3_LOG, HASH2_LOG, FULL_UPDATE> :: clear_hash (BYTE *buf)
{
    mf.clear_hash  (buf);
    clear_hash3 (buf);
}

template <class MatchFinder, int HASH3_LOG, int HASH2_LOG, bool FULL_UPDATE>
void Hash3<MatchFinder, HASH3_LOG, HASH2_LOG, FULL_UPDATE> :: clear_hash3 (BYTE *buf)
{
    if (HTable)  iterate_var(i,HashSize)   HTable[i] =buf+1;
    if (HTable2) iterate_var(i,HashSize2)  HTable2[i]=buf+1;
}

template <class MatchFinder, int HASH3_LOG, int HASH2_LOG, bool FULL_UPDATE>
void Hash3<MatchFinder, HASH3_LOG, HASH2_LOG, FULL_UPDATE> :: shift (BYTE *buf, int shift)
{
    mf.shift (buf, shift);
    iterate_var(i,HashSize)  HTable[i] = HTable[i] >buf+shift? HTable [i]-shift : buf+1;
    iterate_var(i,HashSize2) HTable2[i]= HTable2[i]>buf+shift? HTable2[i]-shift : buf+1;
}



// ****************************************************************************
// This MF just combines any two Match Finders                              ***
//   (mf2 should provide short matches, while mf1 - longer ones)            ***
//   (mf2 isn't searched when mf1 returns match with length > mf1.MINLEN)   ***
// ****************************************************************************

template <class MatchFinder1, class MatchFinder2>
struct CombineMF
{
    MatchFinder1 mf1;
    MatchFinder2 mf2;
    BYTE *q;

    // Constructor just saves references to two matchfinders being combined
    CombineMF (BYTE *buf, uint dictsize, uint hash_size, int hash_row_width, uint auxhash_size, int auxhash_row_width) :
        mf1(buf, dictsize,    hash_size,    hash_row_width, 0, 0),
        mf2(buf, dictsize, auxhash_size, auxhash_row_width, 0, 0)
        {};

    // Prefetch MF data associated with provided `p` (we suppose that mf2 data are small and therefore are already cached)
    void prefetch_hash (BYTE *p)    {mf1.prefetch_hash(p);}

    // Fill the `matches` buffer with len/dist from mf2 (shorter matches), then mf1 (longer matches)
    DISTANCE *find_all_matches (byte *p, void *bufend, DISTANCE *matches)
    {
        matches = mf2.find_all_matches (p, bufend, matches);
        return mf1.find_all_matches (p, bufend, matches);
    }

    // Just try mf1, then mf2 and return best match found
    uint find_matchlen (byte *p, void *bufend, UINT prevlen)
    {
        uint len1 = mf1.find_matchlen (p, bufend, prevlen);
        BYTE *q1  = mf1.get_matchptr();

        if (len1 > mf1.MINLEN) {
            mf2.update_hash1(p);
            q=q1; return len1;
        }

        uint len2 = mf2.find_matchlen (p, bufend, prevlen);
        BYTE *q2  = mf2.get_matchptr();

        // If second match is longer and it's not too far while being just 1 byte longer
        if (len1>=mf1.MINLEN && len1>len2  &&  !(len2>=mf2.MINLEN && len1==len2+1 && ChangePair(p-q2, p-q1)) )
                    {q=q1; return len1;}
        else        {q=q2; return len2;}
    }

    void clear_hash (BYTE *buf)
    {
        mf1.clear_hash (buf);
        mf2.clear_hash (buf);
    }

    void shift (BYTE *buf, int shift)
    {
        mf1.shift (buf, shift);
        mf2.shift (buf, shift);
        q -= shift;
    }

    void update_hash1 (BYTE *p)
    {
        mf1.update_hash1 (p);
        mf2.update_hash1 (p);
    }

    void update_hash (BYTE *p, UINT len, UINT step)
    {
        mf1.update_hash (p, len, step);
        mf2.update_hash (p, len, step);
    }

    void invalidate_match ()
    {
        mf1.invalidate_match();
        mf2.invalidate_match();
    }

    static uint min_length()  {return mymin (MatchFinder1::min_length(), MatchFinder2::min_length());}
    int  error()          {return mymin (mf1.error(), mf2.error());}
    byte *get_matchptr()  {return q;}
};


#undef MINLEN
