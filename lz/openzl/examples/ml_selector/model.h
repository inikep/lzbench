// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <string>

#pragma once

namespace {

const inline std::string_view EXAMPLE_MODEL = R"~DELIM~({
    "predictor": [
        [
            {
                "featureIdx": [
                    -1
                ],
                "value": [
                    0
                ],
                "leftChildIdx": [
                    -1
                ],
                "rightChildIdx": [
                    -1
                ],
                "defaultLeft": [
                    0
                ]
            }
        ]
    ],
    "features": [
        "nbElts"
    ],
    "labels": [
        "fieldlz",
        "fieldlz"
    ]
})~DELIM~";

} // namespace
