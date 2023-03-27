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
 *  Codeplay's SYCL-FFT
 *
 **************************************************************************/

#include <cmath>
#include <gtest/gtest.h>
#include <sycl/sycl.hpp>
#include <vector>

#include "number_generators.hpp"
#include "utils.hpp"

#include "common/bluestein_workitem.hpp"
#include "common/transfers.hpp"

template <int Prime>
void test_bluestein() {
  using ftype = float;
  using vtype = std::complex<ftype>;
  constexpr auto num_elements = Prime;
  constexpr auto padded_len = sycl_fft::detail::next_pow2(2 * num_elements - 1);
  std::vector<vtype> host_input(num_elements);
  std::vector<vtype> host_reference_output(num_elements);
  std::vector<vtype> output(num_elements);

  populate_with_random(host_input, ftype(-1.0), ftype(1.0));

  reference_forward_dft(host_input, host_reference_output, num_elements, 0);

  {
    sycl::queue q(sycl::cpu_selector_v);
    sycl::buffer<std::complex<ftype>, 1> input_buffer(num_elements);
    sycl::buffer<std::complex<ftype>, 1> output_buffer(num_elements);
    sycl_fft::detail::bluestein_data<float> bd(q, num_elements, padded_len);

    q.submit([&](sycl::handler& cgh) {
      auto in_acc = input_buffer.get_access<sycl::access::mode::write>(cgh);
      cgh.copy(host_input.data(), in_acc);
    });

    q.submit([&](sycl::handler& cgh) {
      auto in_acc = input_buffer.get_access<sycl::access::mode::read>(cgh);
      auto mult_acc =
          bd.multipliers_buffer.get_access<sycl::access::mode::read>(cgh);
      auto g_acc = bd.g_buffer.get_access<sycl::access::mode::read>(cgh);
      auto out_acc = output_buffer.get_access<sycl::access::mode::write>(cgh);

      cgh.single_task([=] {
        ftype in[num_elements * 2];
        ftype out[num_elements * 2];
        ftype mult[num_elements * 2];
        ftype g[padded_len * 2];

        for (std::size_t i = 0; i < num_elements; ++i) {
          in[2 * i] = in_acc[i].real();
          in[2 * i + 1] = in_acc[i].imag();
        }
        for (std::size_t i = 0; i < num_elements; ++i) {
          mult[2 * i] = mult_acc[i].real();
          mult[2 * i + 1] = mult_acc[i].imag();
        }
        for (std::size_t i = 0; i < padded_len; ++i) {
          g[2 * i] = g_acc[i].real();
          g[2 * i + 1] = g_acc[i].imag();
        }

        sycl_fft::detail::wi_bluestein_dft<num_elements, padded_len, ftype>(
            in, out, g, mult);

        for (std::size_t i = 0; i < num_elements; ++i) {
          out_acc[i] = {out[2 * i], out[2 * i + 1]};
        }
      });
    });

    q.submit([&](sycl::handler& cgh) {
      auto out_acc = output_buffer.get_access<sycl::access::mode::read>(cgh);
      cgh.copy(out_acc, output.data());
    });
  }
  compare_arrays(output, host_reference_output, 1e-3);
}

template <int... Ps>
void test_bluesteins(std::integer_sequence<int, Ps...>) {
  (test_bluestein<Ps>(), ...);
}

using prime_list = std::integer_sequence<int, 3, 5, 11, 31, 43, 61>;

TEST(Bluestein, HelloTest) { test_bluesteins(prime_list{}); }