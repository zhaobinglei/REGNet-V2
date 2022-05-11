#ifndef _GRIPPER_QUERY
#define _GRIPPER_QUERY

#include <vector>
#include <torch/extension.h>

std::vector<at::Tensor> GripperQuery(
    const at::Tensor points,
    const at::Tensor y_radius,
    const float x_radius,
    const float z_radius,
    const int64_t num_neighbours);

#endif