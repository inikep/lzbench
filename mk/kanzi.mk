# kanzi
CODECS += KANZI
KANZI_OBJS := $(addprefix misc/kanzi-cpp/src/, \
    io/CompressedOutputStream.o io/CompressedInputStream.o entropy/EntropyUtils.o \
    entropy/ExpGolombEncoder.o entropy/FPAQEncoder.o entropy/ANSRangeEncoder.o \
    entropy/ANSRangeDecoder.o entropy/BinaryEntropyDecoder.o entropy/BinaryEntropyEncoder.o \
    entropy/ExpGolombDecoder.o entropy/HuffmanEncoder.o entropy/FPAQDecoder.o \
    entropy/TPAQPredictor.o entropy/CMPredictor.o entropy/HuffmanCommon.o entropy/RangeDecoder.o \
    entropy/RangeEncoder.o entropy/HuffmanDecoder.o bitstream/DefaultInputBitStream.o \
    bitstream/DebugOutputBitStream.o bitstream/DebugInputBitStream.o \
    bitstream/DefaultOutputBitStream.o Event.o Global.o transform/AliasCodec.o transform/BWT.o \
    transform/RLT.o transform/TextCodec.o transform/EXECodec.o transform/SBRT.o \
    transform/ROLZCodec.o transform/LZCodec.o transform/SRT.o transform/DivSufSort.o \
    transform/BWTBlockCodec.o transform/BWTS.o transform/UTFCodec.o transform/ZRLT.o \
    transform/FSDCodec.o)
