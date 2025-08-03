// File: lzham_huffman_codes.h
// LZHAM is in the Public Domain. Please see the Public Domain declaration at the end of include/lzham.h
#pragma once

namespace lzham
{
   //const uint cHuffmanMaxSupportedSyms = 600;
   const uint cHuffmanMaxSupportedSyms = 1024;
   
   uint get_generate_huffman_codes_table_size();
   
   bool generate_huffman_codes(void* pContext, uint num_syms, const uint16* pFreq, uint8* pCodesizes, uint& max_code_size, uint& total_freq_ret);

} // namespace lzham
