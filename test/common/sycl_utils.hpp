/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  Codeplay's portFFT
 *
 **************************************************************************/

#ifndef PORTFFT_TEST_COMMON_SYCL_UTILS_HPP
#define PORTFFT_TEST_COMMON_SYCL_UTILS_HPP

#include <sstream>
#include <string>

#include <sycl/sycl.hpp>

std::string get_device_type(sycl::device dev) {
  using sycl::info::device_type;
  switch (dev.get_info<sycl::info::device::device_type>()) {
    case device_type::cpu:
      return "CPU";
    case device_type::gpu:
      return "GPU";
    case device_type::accelerator:
      return "accelerator";
    case device_type::custom:
      return "custom";
    case device_type::host:
      return "host";
    default:
      return "unknown";
  };
}

std::string get_device_subgroup_sizes(sycl::device dev) {
  auto subgroup_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
  std::stringstream subgroup_sizes_str;
  subgroup_sizes_str << "[";
  for (std::size_t i = 0; i < subgroup_sizes.size(); ++i) {
    if (i > 0) {
      subgroup_sizes_str << ", ";
    }
    subgroup_sizes_str << subgroup_sizes[i];
  }
  subgroup_sizes_str << "]";
  return subgroup_sizes_str.str();
}

#endif  // PORTFFT_TEST_COMMON_SYCL_UTILS_HPP
