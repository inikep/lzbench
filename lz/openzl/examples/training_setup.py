#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys

import parsing_data_generator_correlated as gen

# Sets up a training/ test set using `parsing_data_generator.py`


# The number of samples for training and testing
train_samples = 5
test_samples = 95
random_seed = 0
if __name__ == "__main__":
    # Read directory from args
    DIR = sys.argv[1]

    TRAIN_DIR = os.path.join(DIR, "train")
    TEST_DIR = os.path.join(DIR, "test")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    for _ in range(train_samples):
        filename = os.path.join(TRAIN_DIR, "sample_" + str(random_seed))
        gen.write_formatted_data_to_file(filename, random_seed)
        random_seed += 1
    for _ in range(test_samples):
        filename = os.path.join(TEST_DIR, "sample_" + str(random_seed))
        gen.write_formatted_data_to_file(filename, random_seed)
        random_seed += 1
