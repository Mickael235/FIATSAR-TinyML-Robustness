#pragma once
#include <stddef.h>
#include <stdint.h>

constexpr size_t F1_MODEL_SIZE = 51328;

struct F1TensorOffsetInfo {
  int tensor_index;
  size_t data_offset;
  size_t byte_count;
};

constexpr F1TensorOffsetInfo F1_WEIGHT_TENSORS[] = {
  {4, 1088, 48000},
  {6, 392, 640},
};

constexpr size_t F1_WEIGHT_TENSOR_COUNT =
    sizeof(F1_WEIGHT_TENSORS) / sizeof(F1_WEIGHT_TENSORS[0]);

// Validated reference faults:
// Conv2D w5 b7    : tensor 6, index 5,     offset 397,   -3 -> 125
// Conv2D w31 b7   : tensor 6, index 31,    offset 423,  -20 -> 108
// FC w45024 b7    : tensor 4, index 45024, offset 46112, 48 -> -80
