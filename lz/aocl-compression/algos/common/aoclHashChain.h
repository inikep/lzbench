/**
 * Copyright (C) 2023-2024, Advanced Micro Devices. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from this
 * software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

 /** @file aoclHashChain.h
 *
 *  @brief Implementations of cache efficient hash chains.
 *
 *  This file contains implementations of cache efficient hash chains
 *  that can be used across different algos in AOCL compression library.
 *
 *  @author Ashish Sriram
 */

#ifndef __COMMON_HASH_CHAIN_H
#define __COMMON_HASH_CHAIN_H
typedef size_t chain_t;

/******************************************************************************
* Macro naming convention:
* AOCL_COMMON_MODULE_FUNCTION
* AOCL_COMMON_CEHCFIX_CIRC_INC_HEAD
******************************************************************************/

/****************************************************************************** 
* Cache efficient hash chains (cehc):
* Hash chains are used for collision resolution when a hashing 
* function generates same hashing index (hashIdx) for different inputs.
* Hash chains that are typically implemented using
* linked-lists have poor cache locality.
* 
* Here we implement hash chains with continuously positioned nodes.
* A hash table (hashTable) is used to store the mapping of hashIdx
* to root nodes of hash chains (chainIdx). 
* The hash chains in turn are stored in an array (chainTable).
* The chainIdx points to the hash chain (hashChain) within the chainTable.
* 
* Typical data access pattern would be as follows:
*                                               [hashTable]    
*                                             hashIdx chainIdx
*                                           |        |        |
* input -> [hashing function] -> hashIdx -> |        |        | -> chainIdx
*                                           |        |        |
*
*                                [chainTable]
* chainIdx -> [hashChain_0][hashChain_1]...[hashChain_N] -> hashChain
*
* Begin of common macros for cache efficient hash chains
* Note: Macros do not handle invalid inputs. Calling functions must ensure
*       tables are allocated properly and valid inputs are passed
******************************************************************************/

/******************************************************************************
* Begin of AOCL_COMMON_CEHCFIX:
* cache efficient hash chains with fixed block mapping (CEHCFIX)
* In this implementation of cehc, each hashIdx is mapped to a fixed size
* array in the chainTable that holds its associated hashChain. No separate
* hashTable is needed due to this fixed mapping.
* hashIdx h0 maps to hashChain0, h1 to hashChain1 and so on.
* 
* Params:
* chainTable           : Array of hash chain objects.
*                        Each hash chain object contains a hashChain and 
*                        a head position indicator field (hcHeadPos).
*                        hashChain: hash chain implemented using a circular buffer.
*                        hcHeadPos: position of head of hash chain within the circular buffer.
*                        Format: [hcHeadPos_0|hashChain_0] [hcHeadPos_1|hashChain_1] ... [hcHeadPos_N|hashChain_N]
* hcBase               : Pointer to a hash chain object within chainTable.
* hcCur                : Pointer to nodes within a hash chain object.
* val                  : Value of a node stored within a hash chain.
* hashIdx              : Hash index.
* HASH_CHAIN_MAX       : Max size of hashChain that can be stored within each hash chain object.
* HASH_CHAIN_OBJECT_SZ : Size of each hash chain object. Must be [1 + HASH_CHAIN_MAX].
* kEmptyHeadValue      : Value of empty hcHeadPos.
* kEmptyNodeValue      : Value of empty node in hashChain.
* 
* Typical usage sequence:
* To do an Insert operation, APIs must be called in the following order:
* 1. Call AOCL_COMMON_CEHCFIX_GET_HEAD() to get hcHeadPos.
* 2. Add new node in hashChain at the position next to hcHeadPos 
*    using AOCL_COMMON_CEHCFIX_INSERT(). Also updates hcHeadPos.
* 
* To do a Search operation, APIs must be called in the following order:
* 1. Call AOCL_COMMON_CEHCFIX_GET_HEAD() to get hcHeadPos.
* 2. Call AOCL_COMMON_CEHCFIX_GET() to read value at this position
*    in the hash chain.
* 3. Subsequently, keep calling AOCL_COMMON_CEHCFIX_MOVE_TO_NEXT() 
*    to move to next nodes in the chain and read values there.
******************************************************************************/

// circular buffer position increment operation. hcCur range must be: hcBase+1 <= hcCur <= hcBase+HASH_CHAIN_MAX.
#define AOCL_COMMON_CEHCFIX_CIRC_INC_HEAD(hcCur, HASH_CHAIN_OBJECT_SZ, HASH_CHAIN_MAX) \
((((hcCur + 1) % HASH_CHAIN_OBJECT_SZ) != 0) ? (hcCur + 1) : (hcCur + 1 - HASH_CHAIN_MAX))

// circular buffer position decrement operation. hcCur range must be: hcBase+1 <= hcCur <= hcBase+HASH_CHAIN_MAX.
#define AOCL_COMMON_CEHCFIX_CIRC_DEC_HEAD(hcCur, HASH_CHAIN_OBJECT_SZ, HASH_CHAIN_MAX) \
((((hcCur - 1) % HASH_CHAIN_OBJECT_SZ) != 0) ? (hcCur - 1) : (hcCur + HASH_CHAIN_MAX - 1))

// get head of hash chain
#define AOCL_COMMON_CEHCFIX_GET_HEAD(chainTable, hashTable, hcHeadPos, prevVal, hashIdx, \
    HASH_CHAIN_OBJECT_SZ, HASH_CHAIN_MAX, kEmptyHeadValue) { \
    chain_t hcBase = (chain_t)hashIdx  * HASH_CHAIN_OBJECT_SZ; /* pointer to hash chain object */ \
    hcHeadPos = chainTable[hcBase]; /* read hcHeadPos that is stored at 1st pos of hash chain object */ \
    if (hcHeadPos == kEmptyHeadValue) /* hash chain object is not yet set */ { \
      /* set to 1st node in hashChain. At least 1 node for hcHeadPos and 1 for hashChain must exist */ \
      hcHeadPos = hcBase + 1; \
    } \
}

// set node in hash chain. hcHeadPos must be obtained from AOCL_COMMON_CEHCFIX_GET_HEAD().
#define AOCL_COMMON_CEHCFIX_INSERT(chainTable, hashTable, hcHeadPos, prevVal, val, hashIdx, \
    HASH_CHAIN_OBJECT_SZ, HASH_CHAIN_MAX) { \
    chain_t hcBase = (chain_t)hashIdx  * HASH_CHAIN_OBJECT_SZ; /* pointer to hash chain object */ \
    hcHeadPos = AOCL_COMMON_CEHCFIX_CIRC_DEC_HEAD(hcHeadPos, HASH_CHAIN_OBJECT_SZ, HASH_CHAIN_MAX); /* move to next node in the chain */ \
    chainTable[hcHeadPos] = val; /* set val at new node */ \
    chainTable[hcBase] = hcHeadPos; /* update hcHeadPos to point to the new node */ \
}

// get val at node in hash chain
#define AOCL_COMMON_CEHCFIX_GET(chainTable, hashTable, hcCur, prevVal, val, \
    HASH_CHAIN_OBJECT_SZ, HASH_CHAIN_MAX) \
    val = chainTable[hcCur];

// move to next node in hash chain
#define AOCL_COMMON_CEHCFIX_MOVE_TO_NEXT(chainTable, hcCur, val, \
    HASH_CHAIN_OBJECT_SZ, HASH_CHAIN_MAX, kEmptyNodeValue, hcHeadPos) \
    /* move to next node in chain. Next node position is consecutive */ \
    hcCur = AOCL_COMMON_CEHCFIX_CIRC_INC_HEAD(hcCur, HASH_CHAIN_OBJECT_SZ, HASH_CHAIN_MAX); \
    if (hcCur == hcHeadPos) /* check if one loop in circular buffer got completed */ \
        break; \
    val = chainTable[hcCur]; \
    if (val == kEmptyNodeValue) /* no entry at this node */ \
        break; /* empty node. i.e. end of chain */

/******************************************************************************
* End of AOCL_COMMON_CEHCFIX
******************************************************************************/

/******************************************************************************
* End of cache efficient hash chains
******************************************************************************/

#endif /* __COMMON_HASH_CHAIN_H */
