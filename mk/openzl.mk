# openzl
CODECS += OPENZL

# OpenZL requires a 64-bit platform (src/openzl/shared/portability.h emits an
# #error on 32-bit). Probe the pointer size with the active CODE_FLAGS so this
# also catches -m32 (BUILD_ARCH=32-bit) builds, not just native 32-bit targets.
ifneq ($(shell echo|$(CC) $(CODE_FLAGS) -dM -E - 2>/dev/null|grep -c '__SIZEOF_POINTER__ 8'), 1)
    DONT_BUILD_OPENZL ?= 1
endif

OPENZL_OBJS := $(addprefix lz+entropy/openzl/src/openzl/, \
    codecs/bitSplit/common_bitSplit_kernel.o codecs/bitSplit/decode_bitSplit_binding.o \
    codecs/bitSplit/decode_bitSplit_kernel.o codecs/bitSplit/encode_bitSplit_binding.o \
    codecs/bitSplit/encode_bitSplit_kernel.o codecs/bitSplit/encode_bitsplit_bf16_binding.o \
    codecs/bitSplit/encode_bitsplit_fp_binding.o codecs/bitSplit/encode_bitsplit_top8_binding.o \
    codecs/bitpack/common_bitpack_kernel.o codecs/bitpack/decode_bitpack_binding.o \
    codecs/bitpack/encode_bitpack_binding.o codecs/bitunpack/decode_bitunpack_binding.o \
    codecs/bitunpack/encode_bitunpack_binding.o codecs/common/fast_table.o \
    codecs/common/fast_table16.o codecs/common/fast_tag_table.o codecs/common/window.o \
    codecs/concat/decode_concat_binding.o codecs/concat/encode_concat_binding.o \
    codecs/constant/decode_constant_binding.o codecs/constant/decode_constant_kernel.o \
    codecs/constant/encode_constant_binding.o codecs/constant/encode_constant_kernel.o \
    codecs/conversion/decode_conversion_binding.o codecs/conversion/encode_conversion_binding.o \
    codecs/conversion/encode_setStringSizes_binding.o codecs/conversion/graph_conversion.o \
    codecs/decoder_registry.o codecs/dedup/decode_dedup_binding.o \
    codecs/dedup/encode_dedup_binding.o codecs/delta/decode_delta_binding.o \
    codecs/delta/decode_delta_kernel.o codecs/delta/encode_delta_binding.o \
    codecs/delta/encode_delta_kernel.o codecs/dispatchN_byTag/decode_dispatchN_byTag_binding.o \
    codecs/dispatchN_byTag/decode_dispatchN_byTag_kernel.o \
    codecs/dispatchN_byTag/encode_dispatchN_byTag_binding.o \
    codecs/dispatchN_byTag/encode_dispatchN_byTag_kernel.o \
    codecs/dispatch_by_tag/decode_dispatch_by_tag_kernel.o \
    codecs/dispatch_by_tag/encode_dispatch_by_tag_kernel.o \
    codecs/dispatch_string/decode_dispatch_string_binding.o \
    codecs/dispatch_string/decode_dispatch_string_kernel.o \
    codecs/dispatch_string/encode_dispatch_string_binding.o \
    codecs/dispatch_string/encode_dispatch_string_kernel.o \
    codecs/divide_by/decode_divide_by_binding.o codecs/divide_by/decode_divide_by_kernel.o \
    codecs/divide_by/encode_divide_by_binding.o codecs/divide_by/encode_divide_by_kernel.o \
    codecs/encoder_registry.o codecs/entropy/decode_entropy_binding.o \
    codecs/entropy/decode_huffman_kernel.o codecs/entropy/deprecated/decode_entropy_decompress.o \
    codecs/entropy/deprecated/decode_fse_kernel.o \
    codecs/entropy/deprecated/decode_huf_avx2_decompress.o \
    codecs/entropy/deprecated/encode_entropy_compress.o \
    codecs/entropy/deprecated/encode_fse_kernel.o \
    codecs/entropy/deprecated/encode_huf_avx2_compress.o codecs/entropy/encode_entropy_binding.o \
    codecs/entropy/encode_entropy_selector.o codecs/entropy/encode_huffman_kernel.o \
    codecs/flatpack/decode_flatpack_binding.o codecs/flatpack/decode_flatpack_kernel.o \
    codecs/flatpack/encode_flatpack_binding.o codecs/flatpack/encode_flatpack_kernel.o \
    codecs/float_deconstruct/decode_float_deconstruct_binding.o \
    codecs/float_deconstruct/decode_float_deconstruct_kernel.o \
    codecs/float_deconstruct/encode_float_deconstruct_binding.o \
    codecs/float_deconstruct/encode_float_deconstruct_kernel.o \
    codecs/interleave/decode_interleave_binding.o codecs/interleave/encode_interleave_binding.o \
    codecs/lz/decode_field_lz.o codecs/lz/decode_lz_binding.o codecs/lz/decode_lz_kernel.o \
    codecs/lz/encode_field_lz.o codecs/lz/encode_field_lz_literals_selector.o \
    codecs/lz/encode_field_lz_sequences.o codecs/lz/encode_lz_binding.o \
    codecs/lz/encode_lz_kernel.o codecs/lz/encode_match_finder_fast_field_lz.o \
    codecs/lz/encode_match_finder_greedy_field_lz.o codecs/lz4/decode_lz4_binding.o \
    codecs/lz4/encode_lz4_binding.o codecs/merge_sorted/decode_merge_sorted_binding.o \
    codecs/merge_sorted/decode_merge_sorted_kernel.o \
    codecs/merge_sorted/encode_merge_sorted_binding.o \
    codecs/merge_sorted/encode_merge_sorted_kernel.o \
    codecs/mux_lengths/decode_mux_lengths_binding.o codecs/mux_lengths/decode_mux_lengths_kernel.o \
    codecs/mux_lengths/encode_mux_lengths_binding.o codecs/mux_lengths/encode_mux_lengths_kernel.o \
    codecs/parse_int/decode_parse_int_binding.o codecs/parse_int/decode_parse_int_kernel.o \
    codecs/parse_int/encode_parse_int_binding.o codecs/parse_int/encode_parse_int_kernel.o \
    codecs/partition/common_partition.o codecs/partition/decode_partition_binding.o \
    codecs/partition/decode_partition_bitpack_fusion.o codecs/partition/decode_partition_kernel.o \
    codecs/partition/encode_partition_binding.o codecs/partition/encode_partition_bitpack.o \
    codecs/partition/encode_partition_kernel.o codecs/pivco_huffman/arch/decode_pivco_arch.o \
    codecs/pivco_huffman/arch/decode_pivco_avx512.o codecs/pivco_huffman/arch/encode_pivco_arch.o \
    codecs/pivco_huffman/arch/encode_pivco_avx512.o codecs/pivco_huffman/common_pivco_kernel.o \
    codecs/pivco_huffman/decode_pivco_binding.o codecs/pivco_huffman/decode_pivco_kernel.o \
    codecs/pivco_huffman/encode_pivco_binding.o codecs/pivco_huffman/encode_pivco_kernel.o \
    codecs/prefix/decode_prefix_binding.o codecs/prefix/decode_prefix_kernel.o \
    codecs/prefix/encode_prefix_binding.o codecs/prefix/encode_prefix_kernel.o \
    codecs/quantize/common_quantize.o codecs/quantize/decode_quantize_binding.o \
    codecs/quantize/decode_quantize_kernel.o codecs/quantize/encode_quantize_binding.o \
    codecs/quantize/encode_quantize_kernel.o codecs/range_pack/decode_range_pack_binding.o \
    codecs/range_pack/decode_range_pack_kernel.o codecs/range_pack/encode_range_pack_binding.o \
    codecs/range_pack/encode_range_pack_kernel.o codecs/rolz/decode_experimental_dec.o \
    codecs/rolz/decode_fast_dec.o codecs/rolz/decode_rolz_binding.o \
    codecs/rolz/decode_rolz_kernel.o codecs/rolz/encode_experimental_enc.o \
    codecs/rolz/encode_fast_enc.o codecs/rolz/encode_match_finder_double_fast_lc.o \
    codecs/rolz/encode_match_finder_lazy.o codecs/rolz/encode_rolz_binding.o \
    codecs/rolz/encode_rolz_kernel.o codecs/rolz/encode_rolz_sequences.o \
    codecs/sentinel/decode_sentinel_binding.o codecs/sentinel/decode_sentinel_kernel.o \
    codecs/sentinel/encode_sentinel_binding.o codecs/sentinel/encode_sentinel_kernel.o \
    codecs/sparse_num/decode_sparse_num_binding.o codecs/sparse_num/decode_sparse_num_kernel.o \
    codecs/sparse_num/encode_sparse_num_binding.o codecs/sparse_num/encode_sparse_num_kernel.o \
    codecs/splitByStruct/decode_splitByStruct_binding.o \
    codecs/splitByStruct/decode_splitByStruct_kernel.o \
    codecs/splitByStruct/encode_splitByStruct_binding.o \
    codecs/splitByStruct/encode_splitByStruct_kernel.o codecs/splitN/decode_splitN_binding.o \
    codecs/splitN/decode_splitN_kernel.o codecs/splitN/encode_splitN_binding.o \
    codecs/splitN/encode_split_byrange_binding.o codecs/tokenize/decode_tokenize2to1_kernel.o \
    codecs/tokenize/decode_tokenize4to2_kernel.o codecs/tokenize/decode_tokenizeVarto4_kernel.o \
    codecs/tokenize/decode_tokenize_binding.o codecs/tokenize/decode_tokenize_kernel.o \
    codecs/tokenize/encode_tokenize2to1_kernel.o codecs/tokenize/encode_tokenize4to2_kernel.o \
    codecs/tokenize/encode_tokenizeVarto4_kernel.o codecs/tokenize/encode_tokenize_binding.o \
    codecs/tokenize/encode_tokenize_kernel.o codecs/tokenize/encode_tokenize_kernel_sort.o \
    codecs/transpose/decode_transpose_binding.o codecs/transpose/decode_transpose_kernel.o \
    codecs/transpose/encode_transpose_binding.o codecs/transpose/encode_transpose_kernel.o \
    codecs/zigzag/decode_zigzag_binding.o codecs/zigzag/decode_zigzag_kernel.o \
    codecs/zigzag/encode_zigzag_binding.o codecs/zigzag/encode_zigzag_kernel.o \
    codecs/zstd/common_zstd.o codecs/zstd/decode_zstd_binding.o codecs/zstd/encode_zstd_binding.o \
    common/a1cbor_helpers.o common/allocation.o common/errors.o common/limits.o common/logging.o \
    common/materializer_ctx.o common/opaque.o common/operation_context.o common/refcount.o \
    common/sha256.o common/stream.o common/unique_id.o common/wire_format.o compress/cctx.o \
    compress/cdictmgr.o compress/cgraph.o compress/cnode.o compress/cnodes.o compress/compress2.o \
    compress/compressor_serialization.o compress/dyngraph_interface.o compress/enc_interface.o \
    compress/encode_frameheader.o compress/gcparams.o compress/graph_registry.o compress/graphmgr.o \
    compress/graphs/automated_compressor_explorer.o compress/graphs/generic_clustering_graph.o \
    compress/graphs/sddl/simple_data_description_language.o \
    compress/graphs/sddl/simple_data_description_language_source_code.o \
    compress/graphs/sddl2/sddl2.o compress/graphs/sddl2/sddl2_disasm.o \
    compress/graphs/sddl2/sddl2_interpreter.o compress/graphs/sddl2/sddl2_vm.o \
    compress/graphs/small_lengths_graph.o compress/graphs/split_graph.o \
    compress/implicit_conversion.o compress/localparams.o compress/name.o compress/nodemgr.o \
    compress/rtgraphs.o compress/segmenter.o compress/segmenters/segmenter_numeric.o \
    compress/segmenters/segmenter_serial.o compress/selector.o compress/selectors/ml/features.o \
    compress/selectors/ml/gbt.o compress/selectors/ml/ml_selector_graph.o \
    compress/selectors/ml/mlselector.o compress/selectors/ml/selector_numeric_model.o \
    compress/selectors/selector_brute_force.o compress/selectors/selector_compress.o \
    compress/selectors/selector_constant.o compress/selectors/selector_genericLZ.o \
    compress/selectors/selector_numeric.o compress/selectors/selector_store.o compress/trStates.o \
    decompress/decode_frameheader.o decompress/decoder_fusion.o decompress/decompress2.o \
    decompress/dictx.o decompress/dtransforms.o decompress/gdparams.o decompress/reflection.o \
    dict/bundle.o dict/dict.o dict/dictloader.o dict/fatbundle_dictloader.o fse/common/debug.o \
    fse/common/entropy_common.o fse/common/error_private.o fse/compress/fse_compress.o \
    fse/compress/hist.o fse/compress/huf_compress.o fse/decompress/fse_decompress.o \
    fse/decompress/huf_decompress.o shared/a1cbor.o shared/base64.o shared/clustering_common.o \
    shared/clustering_compress.o shared/data_stats.o shared/detail/pdqsort1.o \
    shared/detail/pdqsort2.o shared/detail/pdqsort4.o shared/detail/pdqsort8.o shared/estimate.o \
    shared/histogram.o shared/numeric_operations.o)
# assembled from huf_decompress_amd64.S
OPENZL_OBJS  += lz+entropy/openzl/src/openzl/fse/decompress/huf_decompress_amd64.o
OPENZL_FLAGS := -I$(SRC)lz+entropy/zstd/lib -I$(SRC)lz/lz4/lib -I$(SRC)lz+entropy/openzl/include -I$(SRC)lz+entropy/openzl/src
