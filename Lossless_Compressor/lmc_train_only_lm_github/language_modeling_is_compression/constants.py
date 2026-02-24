# Copyright 2024 DeepMind Technologies Limited
# Modifications Copyright 2026 Zhejiang University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0

"""Defines project-wide constants."""


NUM_CHUNKS = 100  # Default number of chunks to process in chunked datasets.
CHUNK_SIZE_BYTES = 78  # Size of each chunk in bytes for chunked datasets.
CHUNK_SHAPE_2D = (32, 64)
ALPHABET_SIZE = 256

# Base 2 means that the coder writes bits.
ARITHMETIC_CODER_BASE = 2
# Precision 32 implies 32 bit arithmetic.
ARITHMETIC_CODER_PRECISION = 32
