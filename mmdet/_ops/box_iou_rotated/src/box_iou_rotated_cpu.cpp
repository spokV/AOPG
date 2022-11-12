// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include <torch/types.h>
#include "box_iou_rotated_utils.h"


template <typename T>
void box_iou_rotated_cpu_kernel(
    const at::Tensor& boxes1,
    const at::Tensor& boxes2,
    const bool iou_or_iof,
    at::Tensor& ious) {
  auto num_boxes1 = boxes1.size(0);
  auto num_boxes2 = boxes2.size(0);

  for (int i = 0; i < num_boxes1; i++) {
    for (int j = 0; j < num_boxes2; j++) {
      ious[i * num_boxes2 + j] = single_box_iou_rotated<T>(
          boxes1[i].data_ptr<T>(), boxes2[j].data_ptr<T>(), iou_or_iof);
    }
  }
}

at::Tensor box_iou_rotated_cpu(
    // input must be contiguous:
    const at::Tensor& boxes1,
    const at::Tensor& boxes2,
    const bool iou_or_iof) {
  auto num_boxes1 = boxes1.size(0);
  auto num_boxes2 = boxes2.size(0);
  at::Tensor ious =
      at::empty({num_boxes1 * num_boxes2}, boxes1.options().dtype(at::kFloat));

  box_iou_rotated_cpu_kernel<float>(boxes1, boxes2, iou_or_iof, ious);

  // reshape from 1d array to 2d array
  auto shape = std::vector<int64_t>{num_boxes1, num_boxes2};
  return ious.reshape(shape);
}
