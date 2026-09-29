# brotli
CODECS += BROTLI
BROTLI_OBJS := $(addprefix lz+entropy/brotli/, \
    common/constants.o common/context.o common/dictionary.o common/platform.o common/transform.o \
    dec/bit_reader.o dec/decode.o dec/huffman.o dec/prefix.o dec/state.o dec/static_init.o \
    enc/backward_references.o enc/block_splitter.o enc/brotli_bit_stream.o enc/encode.o \
    enc/encoder_dict.o enc/entropy_encode.o enc/fast_log.o enc/histogram.o enc/command.o \
    enc/literal_cost.o enc/memory.o enc/metablock.o enc/static_dict.o enc/static_dict_lut.o \
    enc/static_init.o enc/utf8_util.o enc/compress_fragment.o enc/compress_fragment_two_pass.o \
    enc/cluster.o enc/bit_cost.o enc/backward_references_hq.o enc/dictionary_hash.o \
    common/shared_dictionary.o enc/compound_dictionary.o)
BROTLI_FLAGS := -I$(SRC)lz+entropy/brotli/include
