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

#include <sycl.hpp>
#include <iostream>

#include <benchmark/benchmark.h>
#include "number_generators.hpp"
#include <common/bluestein_workitem.hpp>
#include <common/transfers.hpp>

constexpr int N = 13;
constexpr int sg_size = 16;
constexpr int n_sgs = 100;
constexpr int n_transforms = sg_size * n_sgs;
using ftype = float;
using complex_type = std::complex<ftype>;

template <int N, int N_padded, typename T>
void __attribute__((noinline)) dft_wrapper(T* in_out, T* g, T* multipliers) {
  sycl_fft::detail::wi_bluestein_dft<N, N_padded, T>(in_out, in_out, g, multipliers);
}

constexpr static sycl::specialization_id<int> size_spec_const;

static void BM_dft(benchmark::State& state) {
  constexpr size_t elements_per_subgroup = N * sg_size;
  constexpr size_t elements = elements_per_subgroup * n_sgs;
  constexpr auto padded_N = sycl_fft::detail::next_pow2(2*N-1);
  std::vector<complex_type> a(elements);
  std::array<complex_type, elements> b;
  populate_with_random(a);

  sycl::queue q;
  sycl::buffer<complex_type,1> a_dev(elements);
  sycl::buffer<complex_type,1> b_dev(elements);
  q.submit([&](sycl::handler& cgh) {
    auto a_acc = a_dev.get_access<sycl::access::mode::write>(cgh);
    cgh.copy(a.data(), a_acc);
  });
  sycl_fft::detail::bluestein_data<ftype> bdata(q, N, padded_N);

  q.wait();

  auto run = [&]() {
    q.submit([&](sycl::handler& h) {
       h.set_specialization_constant<size_spec_const>(N);
       auto a_acc = a_dev.get_access<sycl::access::mode::read>(h);
       auto b_acc = b_dev.get_access<sycl::access::mode::write>(h);
       auto mult_acc = bdata.multipliers_buffer.get_access<sycl::access::mode::read>(h);
       auto g_acc = bdata.g_buffer.get_access<sycl::access::mode::read>(h);
       sycl::local_accessor<complex_type, 1> loc(elements_per_subgroup + N + padded_N, h);
       h.parallel_for(
           sycl::nd_range<1>({sg_size * n_sgs}, {sg_size}),
           [=](sycl::nd_item<1> it, sycl::kernel_handler kh)
               [[intel::reqd_sub_group_size(sg_size)]] {
                 int Nn = kh.get_specialization_constant<size_spec_const>();
                 sycl::sub_group sg = it.get_sub_group();
                 size_t local_id = sg.get_local_linear_id();

                 complex_type priv[N];
                 complex_type multipliers[N];
                 complex_type g[padded_N];

                 sycl_fft::global2local(a_acc, loc, elements_per_subgroup, sg_size,
                                        local_id);
                 sycl_fft::global2local(mult_acc, loc, N, N,local_id,0,elements_per_subgroup);
                 sycl_fft::global2local(g_acc, loc, padded_N, padded_N,local_id,0,elements_per_subgroup + N);

                 sycl::group_barrier(sg);

                 sycl_fft::local2private<N>(loc, priv, local_id, sg_size, 0);
                 sycl_fft::local2private<N>(loc, multipliers, local_id, sg_size, elements_per_subgroup);
                 sycl_fft::local2private<N>(loc, g, local_id, sg_size, elements_per_subgroup+N);

                 for (long j = 0; j < 10; j++) {
                   dft_wrapper<N, padded_N>(reinterpret_cast<ftype*>(priv),
                                  reinterpret_cast<ftype*>(g),
                                  reinterpret_cast<ftype*>(multipliers));
                 }
                 sycl_fft::private2local<N>(priv, loc, local_id, sg_size);
                 sycl::group_barrier(sg);
                 sycl_fft::local2global(loc, b_acc, N * sg_size, sg_size,
                                        local_id);
               });
     }).wait();
  };

  // warmup
  run();

  for (auto _ : state) {
    run();
  }
}

BENCHMARK(BM_dft);

BENCHMARK_MAIN();
