// Copyright (c) Meta Platforms, Inc. and affiliates.

// This header defines constants which control the following properties of the
// Thrift parser:
//
// - The set of singleton outcomes: SingletonId
// - The set of variable outcomes: VariableOutcome
// - The set of singleton streams: SingletonId
// - The set of variable streams: LogicalId
// - The set of special Thrift node values: ThriftNodeId

#pragma once

#include "custom_transforms/thrift/thrift_parsers.h" // @manual
#include "openzl/shared/portability.h"               // @manual
#include "openzl/zl_data.h"                          // @manual

#include <folly/Conv.h>
#include <folly/Range.h>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <variant>
#include <vector>

namespace zstrong::thrift {

enum class SingletonId {
    kTypes = 0,
    kFieldDeltas,
    kLengths,
    kBool,
    kInt8,
    kInt16,
    kInt32,
    kInt64,
    kFloat32,
    kFloat64,
    kBinary,
    kConfig,
    kNumSingletonIds
};
enum class LogicalId : uint16_t {};

using StreamId = std::variant<SingletonId, LogicalId>;

enum class VariableOutcome {
    kSerialized = 0,
    kNumeric,
    kVSF,
    kClusterSegmentLengths,
    kNumVariableOutcomes
};

enum class ThriftNodeId : int32_t {
    kMapKey        = std::numeric_limits<int32_t>::max(),
    kMapValue      = std::numeric_limits<int32_t>::max() - 1,
    kListElem      = std::numeric_limits<int32_t>::max() - 2,
    kStop          = std::numeric_limits<int32_t>::max() - 3,
    kRoot          = std::numeric_limits<int32_t>::max() - 4,
    kLength        = std::numeric_limits<int32_t>::max() - 5,
    kMessageHeader = std::numeric_limits<int32_t>::max() - 6,
};
using ThriftPath     = std::vector<ThriftNodeId>;
using ThriftPathView = folly::Range<const ThriftNodeId*>;

ZL_INLINE bool validateThriftNodeId(ThriftNodeId id, int minFormatVersion)
{
    switch (id) {
        case ThriftNodeId::kMapKey:
        case ThriftNodeId::kMapValue:
        case ThriftNodeId::kListElem:
        case ThriftNodeId::kRoot:
        case ThriftNodeId::kLength:
            return minFormatVersion >= kMinFormatVersionEncode;
        case ThriftNodeId::kMessageHeader:
            return minFormatVersion >= kMinFormatVersionEncodeTulipV2;
        default:
            return false;
    }
}

// TODO(T171816129) Specialize folly::hasher for ThriftPath and ThriftPathView
// so we can use fast hash tables

ZL_INLINE bool isSpecialId(ThriftNodeId id)
{
    return static_cast<int32_t>(id) < std::numeric_limits<int16_t>::min()
            || static_cast<int32_t>(id) > std::numeric_limits<int16_t>::max();
}

struct OutcomeInfo {
    ZL_Type type;
    size_t idx;
};

constexpr OutcomeInfo getOutcomeInfo(SingletonId outcome)
{
    const size_t idx = (size_t)outcome;
    switch (outcome) {
        case SingletonId::kTypes:
        case SingletonId::kBool:
        case SingletonId::kInt8:
        case SingletonId::kBinary:
        case SingletonId::kConfig:
            return { .type = ZL_Type_serial, .idx = idx };
        case SingletonId::kFieldDeltas:
        case SingletonId::kInt16:
        case SingletonId::kLengths:
        case SingletonId::kInt32:
        case SingletonId::kFloat32:
        case SingletonId::kInt64:
        case SingletonId::kFloat64:
            return { .type = ZL_Type_numeric, .idx = idx };
        case SingletonId::kNumSingletonIds:
        default:
            throw std::runtime_error{
                "kNumSingletonIds is not a valid outcome!"
            };
    };
}

constexpr OutcomeInfo getOutcomeInfo(VariableOutcome outcome)
{
    const size_t baseIdx = (size_t)SingletonId::kNumSingletonIds;
    switch (outcome) {
        case VariableOutcome::kSerialized:
            return { .type = ZL_Type_serial, .idx = baseIdx };
        case VariableOutcome::kNumeric:
            return { .type = ZL_Type_numeric, .idx = baseIdx + 1 };
        case VariableOutcome::kVSF:
            return { .type = ZL_Type_string, .idx = baseIdx + 2 };
        case VariableOutcome::kClusterSegmentLengths:
            return { .type = ZL_Type_numeric, .idx = baseIdx + 3 };
        case VariableOutcome::kNumVariableOutcomes:
        default:
            throw std::runtime_error{
                "kNumVariableOutcomes is not a valid outcome!"
            };
    }
}

// Maximum decoding expansion factor for TCompact / TBinary
constexpr size_t kMaxExpansionFactor = 11;

inline std::string pathToStr(const ThriftPath& path)
{
    std::string s;
    for (const auto id : path) {
        if (s.empty()) {
            s += "[";
        } else {
            s += ", ";
        }
        s += folly::to<std::string>(id);
    }
    if (s.empty()) {
        s += "[";
    }
    s += "]";
    return s;
}

} // namespace zstrong::thrift
