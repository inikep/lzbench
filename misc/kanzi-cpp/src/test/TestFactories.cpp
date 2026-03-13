/*
Copyright 2011-2026 Frederic Langlet
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

                http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include "../Context.hpp"
#include "../bitstream/DefaultInputBitStream.hpp"
#include "../bitstream/DefaultOutputBitStream.hpp"
#include "../entropy/EntropyDecoderFactory.hpp"
#include "../entropy/EntropyEncoderFactory.hpp"
#include "../transform/TransformFactory.hpp"

using namespace std;
using namespace kanzi;

#define ASSERT_TRUE(cond, msg)                                           \
    do {                                                                 \
        if (!(cond)) {                                                   \
            cerr << "ASSERT FAILED: " << msg << " (" << __FILE__         \
                 << ":" << __LINE__ << ")" << endl;                      \
            return 1;                                                    \
        }                                                                \
    } while (0)

static bool expectInvalidArgument(void (*fn)())
{
    try {
        fn();
    }
    catch (const invalid_argument&) {
        return true;
    }

    return false;
}

static void callUnknownTransformType()
{
    TransformFactory<kanzi::byte>::getType("not-a-transform");
}

static void callTooManyTransforms()
{
    TransformFactory<kanzi::byte>::getType("LZ+RLT+TEXT+UTF+EXE+PACK+DNA+MM+SRT");
}

static void callUnknownEncoderType()
{
    EntropyEncoderFactory::getType("bad-entropy");
}

static void callUnknownDecoderType()
{
    EntropyDecoderFactory::getType("bad-entropy");
}

static void callUnknownEncoderName()
{
    EntropyEncoderFactory::getName(42);
}

static void callUnknownDecoderName()
{
    EntropyDecoderFactory::getName(42);
}

static int testTransformFactory()
{
    cout << "Test TransformFactory" << endl;

    ASSERT_TRUE(TransformFactory<kanzi::byte>::getTypeToken("text") == TransformFactory<kanzi::byte>::DICT_TYPE,
        "TEXT transform must be case insensitive");
    ASSERT_TRUE(TransformFactory<kanzi::byte>::getType("NONE") == 0,
        "NONE transform must encode to zero");
    ASSERT_TRUE(TransformFactory<kanzi::byte>::getType("NONE+NONE") == 0,
        "Null transforms must be skipped");
    ASSERT_TRUE(TransformFactory<kanzi::byte>::getName(
        TransformFactory<kanzi::byte>::getType("LZ+NONE+RLT")) == "LZ+RLT",
        "Transform names must round-trip without null transforms");
    ASSERT_TRUE(expectInvalidArgument(callUnknownTransformType),
        "Unknown transform type must throw");
    ASSERT_TRUE(expectInvalidArgument(callTooManyTransforms),
        "More than eight transforms must throw");

    {
        Context ctx;
        ctx.putString("entropy", "NONE");
        TransformSequence<kanzi::byte>* seq = TransformFactory<kanzi::byte>::newTransform(ctx,
            TransformFactory<kanzi::byte>::getType("TEXT"));
        ASSERT_TRUE(seq->getNbTransforms() == 1, "TEXT sequence must contain one transform");
        ASSERT_TRUE(ctx.getInt("textcodec", 0) == 2, "TEXT must select codec 2 with NONE entropy");
        delete seq;
    }

    {
        Context ctx;
        ctx.putString("entropy", "FPAQ");
        TransformSequence<kanzi::byte>* seq = TransformFactory<kanzi::byte>::newTransform(ctx,
            TransformFactory<kanzi::byte>::getType("TEXT"));
        ASSERT_TRUE(ctx.getInt("textcodec", 0) == 1, "TEXT must select codec 1 with FPAQ entropy");
        delete seq;
    }

    {
        Context ctx;
        TransformSequence<kanzi::byte>* seq = TransformFactory<kanzi::byte>::newTransform(ctx,
            TransformFactory<kanzi::byte>::getType("LZX"));
        ASSERT_TRUE(ctx.getInt("lz", 0) == TransformFactory<kanzi::byte>::LZX_TYPE,
            "LZX transform must set lz context");
        delete seq;
    }

    {
        Context ctx;
        TransformSequence<kanzi::byte>* seq = TransformFactory<kanzi::byte>::newTransform(ctx,
            TransformFactory<kanzi::byte>::getType("LZP"));
        ASSERT_TRUE(ctx.getInt("lz", 0) == TransformFactory<kanzi::byte>::LZP_TYPE,
            "LZP transform must set lz context");
        delete seq;
    }

    {
        Context ctx;
        TransformSequence<kanzi::byte>* seq = TransformFactory<kanzi::byte>::newTransform(ctx,
            TransformFactory<kanzi::byte>::getType("DNA"));
        ASSERT_TRUE(ctx.getInt("packOnlyDNA", 0) == 1,
            "DNA transform must set packOnlyDNA context");
        delete seq;
    }

    return 0;
}

static int testEntropyFactories()
{
    cout << "Test Entropy factories" << endl;

    ASSERT_TRUE(EntropyEncoderFactory::getType("none") == EntropyEncoderFactory::NONE_TYPE,
        "NONE entropy encoder must be case insensitive");
    ASSERT_TRUE(EntropyDecoderFactory::getType("ans0") == EntropyDecoderFactory::ANS0_TYPE,
        "ANS0 entropy decoder must be case insensitive");
    ASSERT_TRUE(string(EntropyEncoderFactory::getName(EntropyEncoderFactory::HUFFMAN_TYPE)) == "HUFFMAN",
        "Encoder name must round-trip");
    ASSERT_TRUE(string(EntropyDecoderFactory::getName(EntropyDecoderFactory::TPAQX_TYPE)) == "TPAQX",
        "Decoder name must round-trip");
    ASSERT_TRUE(expectInvalidArgument(callUnknownEncoderType),
        "Unknown encoder type must throw");
    ASSERT_TRUE(expectInvalidArgument(callUnknownDecoderType),
        "Unknown decoder type must throw");
    ASSERT_TRUE(expectInvalidArgument(callUnknownEncoderName),
        "Unknown encoder name must throw");
    ASSERT_TRUE(expectInvalidArgument(callUnknownDecoderName),
        "Unknown decoder name must throw");

    stringbuf buffer;
    iostream io(&buffer);
    DefaultOutputBitStream obs(io, 16384);
    Context ctx;

    EntropyEncoder* encoders[] = {
        EntropyEncoderFactory::newEncoder(obs, ctx, EntropyEncoderFactory::NONE_TYPE),
        EntropyEncoderFactory::newEncoder(obs, ctx, EntropyEncoderFactory::HUFFMAN_TYPE),
        EntropyEncoderFactory::newEncoder(obs, ctx, EntropyEncoderFactory::RANGE_TYPE),
        EntropyEncoderFactory::newEncoder(obs, ctx, EntropyEncoderFactory::ANS0_TYPE),
        EntropyEncoderFactory::newEncoder(obs, ctx, EntropyEncoderFactory::ANS1_TYPE),
        EntropyEncoderFactory::newEncoder(obs, ctx, EntropyEncoderFactory::FPAQ_TYPE),
        EntropyEncoderFactory::newEncoder(obs, ctx, EntropyEncoderFactory::CM_TYPE),
        EntropyEncoderFactory::newEncoder(obs, ctx, EntropyEncoderFactory::TPAQ_TYPE),
        EntropyEncoderFactory::newEncoder(obs, ctx, EntropyEncoderFactory::TPAQX_TYPE)
    };

    for (size_t i = 0; i < sizeof(encoders) / sizeof(encoders[0]); i++) {
        ASSERT_TRUE(encoders[i] != nullptr, "Entropy encoder must be created");
        encoders[i]->dispose();
        delete encoders[i];
    }

    obs.close();
    io.rdbuf()->pubseekpos(0);
    DefaultInputBitStream ibs(io, 16384);

    EntropyDecoder* decoders[] = {
        EntropyDecoderFactory::newDecoder(ibs, ctx, EntropyDecoderFactory::NONE_TYPE),
        EntropyDecoderFactory::newDecoder(ibs, ctx, EntropyDecoderFactory::HUFFMAN_TYPE),
        EntropyDecoderFactory::newDecoder(ibs, ctx, EntropyDecoderFactory::RANGE_TYPE),
        EntropyDecoderFactory::newDecoder(ibs, ctx, EntropyDecoderFactory::ANS0_TYPE),
        EntropyDecoderFactory::newDecoder(ibs, ctx, EntropyDecoderFactory::ANS1_TYPE),
        EntropyDecoderFactory::newDecoder(ibs, ctx, EntropyDecoderFactory::FPAQ_TYPE),
        EntropyDecoderFactory::newDecoder(ibs, ctx, EntropyDecoderFactory::CM_TYPE),
        EntropyDecoderFactory::newDecoder(ibs, ctx, EntropyDecoderFactory::TPAQ_TYPE),
        EntropyDecoderFactory::newDecoder(ibs, ctx, EntropyDecoderFactory::TPAQX_TYPE)
    };

    for (size_t i = 0; i < sizeof(decoders) / sizeof(decoders[0]); i++) {
        ASSERT_TRUE(decoders[i] != nullptr, "Entropy decoder must be created");
        decoders[i]->dispose();
        delete decoders[i];
    }

    ibs.close();
    return 0;
}

int main()
{
    if (testTransformFactory() != 0)
        return 1;

    if (testEntropyFactories() != 0)
        return 1;

    cout << "All factory tests passed." << endl;
    return 0;
}
