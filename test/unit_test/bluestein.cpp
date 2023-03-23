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

#include "common/transfers.hpp"
#include "common/workitem.hpp"

// list of primes less than 64
constexpr std::array<int, 18> primes{2,  3,  5,  7,  11, 13, 17, 19, 23,
                                     29, 31, 37, 41, 43, 47, 53, 59, 61};

template <typename ftype>
void build_multiplier(std::complex<ftype>* m, std::size_t N) {
  for (int k = 0; k != N; k += 1) {
    const double inner =
        (double(-M_PI) * static_cast<double>(k * k)) / static_cast<double>(N);
    m[k] = {static_cast<ftype>(std::cos(inner)),
            static_cast<ftype>(std::sin(inner))};
  }
}

template <typename ftype>
void build_g(std::complex<ftype>* g, std::size_t N_padded, double N) {
  for (std::size_t n = 0; n != N_padded; n += 1) {
    // note that sin(-x) = -sin(x) and cos(-x) = cos(x), so this is connected to
    // build_multiplier
    const double inner = (double(M_PI) * static_cast<double>(n * n)) / N;
    g[n] = {static_cast<ftype>(std::cos(inner)),
            static_cast<ftype>(std::sin(inner))};
    g[N_padded - n] = g[n];
  }
}

TEST(Bluestein, HelloTest) {
  using ftype = float;
  using vtype = std::complex<ftype>;
  constexpr auto num_elements = primes[17];
  constexpr auto padded_len = num_elements * 2;
  std::vector<vtype> host_input(num_elements);
  std::vector<vtype> host_reference_output(num_elements);
  std::vector<vtype> output(num_elements);
  const auto multiplier = std::make_unique<vtype[]>(num_elements);
  build_multiplier(multiplier.get(), num_elements);
  const auto g_host = std::make_unique<vtype[]>(padded_len);
  build_g(g_host.get(), padded_len, static_cast<double>(num_elements));

  for (size_t i = 0; i< num_elements; ++i){
    host_input[i] = vtype{ftype(i),0}; 
  }
  //populate_with_random(host_input, ftype(-1.0), ftype(1.0));
  reference_forward_dft(host_input, host_reference_output, num_elements, 0);
  {
    sycl::queue q(sycl::cpu_selector_v);
    sycl::buffer<std::complex<ftype>, 1> input_buffer(num_elements);
    sycl::buffer<std::complex<ftype>, 1> output_buffer(num_elements);
    sycl::buffer<std::complex<ftype>, 1> mult_buffer(num_elements);
    sycl::buffer<std::complex<ftype>, 1> g_buffer(padded_len);

    q.submit([&](sycl::handler& cgh) {
      auto in_acc = input_buffer.get_access<sycl::access::mode::write>(cgh);
      cgh.copy(host_input.data(), in_acc);
    });

    q.submit([&](sycl::handler& cgh) {
      auto mult_acc = mult_buffer.get_access<sycl::access::mode::write>(cgh);
      cgh.copy(multiplier.get(), mult_acc);
    });

    q.submit([&](sycl::handler& cgh) {
      auto g_acc = g_buffer.get_access<sycl::access::mode::write>(cgh);
      cgh.copy(g_host.get(), g_acc);
    });

    q.submit([&](sycl::handler& cgh) {
      auto in_acc = input_buffer.get_access<sycl::access::mode::read>(cgh);
      auto mult_acc = mult_buffer.get_access<sycl::access::mode::read>(cgh);
      auto g_acc = g_buffer.get_access<sycl::access::mode::read>(cgh);
      auto out_acc = output_buffer.get_access<sycl::access::mode::write>(cgh);

      cgh.single_task<class finlays>([=] {
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

        sycl_fft::wi_bluestein_dft<num_elements, padded_len, ftype>(in, out, g,
                                                                    mult);

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

  ftype max_diff = 0;
  for (std::size_t i = 0; i != num_elements; i += 1) {
    ftype diff = std::abs(output[i] - host_reference_output[i]);
    max_diff = std::max(max_diff, diff);
  }
  std::cout << "maximum diff = " << max_diff << '\n';
}