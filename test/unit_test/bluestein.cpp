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

template <typename Float>
struct bluestein_data {
  sycl::buffer<std::complex<Float>, 1> multipliers_buffer;
  sycl::buffer<std::complex<Float>, 1> g_buffer;

  bluestein_data(sycl::queue q, size_t N, size_t padded_N)
      : multipliers_buffer(N), g_buffer(padded_N) {
    assert(padded_N >= 2 * N - 1);

    auto multipliers = std::make_unique<std::complex<Float>[]>(N);
    for (size_t k = 0; k != N; k += 1) {
      const double inner =
          (double(-M_PI) * static_cast<double>(k * k)) / static_cast<double>(N);
      multipliers[k] = {static_cast<Float>(std::cos(inner)),
                        static_cast<Float>(std::sin(inner))};
    }

    auto m_event = q.submit([&](sycl::handler& cgh) {
      auto mult_acc =
          multipliers_buffer.template get_access<sycl::access::mode::write>(
              cgh);
      cgh.copy(multipliers.get(), mult_acc);
    });

    auto g = std::make_unique<std::complex<Float>[]>(padded_N);
    g[0] = {1, 0};
    for (std::size_t n = 1; n != N; n += 1) {
      // note that sin(-x) = -sin(x) and cos(-x) = cos(x), so this is connected
      // to multiplier
      const double inner =
          (double(M_PI) * static_cast<double>(n * n)) / static_cast<double>(N);
      g[n] = {static_cast<Float>(std::cos(inner)),
              static_cast<Float>(std::sin(inner))};
      g[padded_N - n] = g[n];
    }

    auto g_event = q.submit([&](sycl::handler& cgh) {
      auto g_acc = g_buffer.template get_access<sycl::access::mode::write>(cgh);
      cgh.copy(g.get(), g_acc);
    });

    m_event.wait();
    g_event.wait();
  }
};

template <typename Integer>
inline constexpr Integer next_pow2(Integer n) {
  if constexpr (sizeof(Integer) == sizeof(std::uint32_t)) {
#if __has_builtin(__builtin_clz)
    static_assert(sizeof(std::uint32_t) == sizeof(unsigned int));
    return 1 << (32 - __builtin_clz(static_cast<std::uint32_t>(n)));
#else
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
#endif
  } else if constexpr (sizeof(Integer) == sizeof(std::uint64_t)) {
#if __has_builtin(__builtin_clzl)
    static_assert(sizeof(std::uint64_t) == sizeof(unsigned long));
    return 1 << (64 - __builtin_clzl(static_cast<std::uint64_t>(n)));
#else
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
#endif
  }
}

template <int Prime>
void test_bluestein() {
  using ftype = float;
  using vtype = std::complex<ftype>;
  constexpr auto num_elements = Prime;
  constexpr auto padded_len = next_pow2(2 * num_elements - 1);
  std::vector<vtype> host_input(num_elements);
  std::vector<vtype> host_reference_output(num_elements);
  std::vector<vtype> output(num_elements);

  populate_with_random(host_input, ftype(-1.0), ftype(1.0));

  reference_forward_dft(host_input, host_reference_output, num_elements, 0);

  {
    sycl::queue q(sycl::cpu_selector_v);
    sycl::buffer<std::complex<ftype>, 1> input_buffer(num_elements);
    sycl::buffer<std::complex<ftype>, 1> output_buffer(num_elements);
    bluestein_data<float> bd(q, num_elements, padded_len);

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