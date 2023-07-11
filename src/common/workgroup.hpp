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

#ifndef SYCL_FFT_COMMON_WORKGROUP_HPP
#define SYCL_FFT_COMMON_WORKGROUP_HPP

#include <common/helpers.hpp>
#include <common/subgroup.hpp>
#include <enums.hpp>

namespace sycl_fft {

// The overall plan is to do 4 steps
//  1. M N-sized dfts along the columns
//  2. multiply rows by twiddle factors
//  3. N M-sized dfts along the rows
//  4. transpose the result

// each sub-fft is done with sg_dft (maybe the columns should be done with wi_dft??)

// do M N-sized dfts, each dft is down a column :'(
template <direction Dir, int N, int M, int WIPerFFT, int FFTSizePerWI, int SubgroupSize, typename T>
__attribute__((always_inline)) inline void column_ffts(sycl::sub_group sg, T* priv, T* loc, T* loc_twiddles) {
  constexpr int ffts_per_sg = SubgroupSize / WIPerFFT;

  // if things don't divide equally then some of the work items might not be active in the FFT, but they still need to
  // call sg_dft
  static_assert(M % ffts_per_sg == 0, "need to account for a part of a subgroup breaking from the for-loop");
  static_assert(SubgroupSize % WIPerFFT == 0, "need to account for some work-items being inactive in the sg dft");

  static_assert(WIPerFFT * FFTSizePerWI == N);
  if constexpr (M == 64 && N == 64) {
    static_assert(WIPerFFT == 32);
    static_assert(FFTSizePerWI == 2);
    static_assert(ffts_per_sg == 1);
  }

  // which of the sub-groups ffts will this wi do
  int fft_in_sg = static_cast<int>(sg.get_local_linear_id()) / WIPerFFT;
  const int fft_local_id = static_cast<int>(sg.get_local_linear_id()) % WIPerFFT;

  constexpr int column_step = SYCLFFT_SGS_IN_WG * ffts_per_sg;
  int column_begin = static_cast<int>(sg.get_group_id()) * ffts_per_sg + fft_in_sg;

  for (int column = column_begin; column < M; column += column_step) {
    local2private_transposed<FFTSizePerWI, M, detail::pad::DO_PAD>(loc, priv, fft_local_id, column);
    sg_dft<Dir, FFTSizePerWI, WIPerFFT>(priv, sg, loc_twiddles + (2 * M));
    private2local_transposed<FFTSizePerWI, M, detail::pad::DO_PAD>(priv, loc, fft_local_id, WIPerFFT, column);
  }
}

// do N M-sized dfts, each dft is along a row :D
template <direction Dir, int N, int M, int WIPerFFT, int FFTSizePerWI, int SubgroupSize, typename T>
__attribute__((always_inline)) inline void row_ffts(sycl::sub_group sg, T* priv, T* loc, T* loc_twiddles,
                                                    const T* wg_twiddles, T scaling_factor) {
  constexpr int ffts_per_sg = SubgroupSize / WIPerFFT;
  // if things don't divide equally then some of the work items might not be active in the FFT, but they still need to
  // call sg_dft
  static_assert(N % ffts_per_sg == 0, "need to account for a part of a subgroup breaking from the for-loop");
  static_assert(SubgroupSize % WIPerFFT == 0, "need to account for some work-items being inactive in the sg dft");
  if constexpr (M == 64 && N == 64) {
    static_assert(WIPerFFT == 32);
    static_assert(FFTSizePerWI == 2);
    static_assert(ffts_per_sg == 1);
  }

  const int fft_in_sg = static_cast<int>(sg.get_local_linear_id()) / WIPerFFT;
  const std::size_t fft_local_id = sg.get_local_linear_id() % WIPerFFT;

  constexpr int row_step = SYCLFFT_SGS_IN_WG * ffts_per_sg;
  const int row_begin = static_cast<int>(sg.get_group_id()) * ffts_per_sg + fft_in_sg;

  for (int row = row_begin; row < N; row += row_step) {
    local2private<2 * FFTSizePerWI, detail::pad::DO_PAD>(loc, priv, fft_local_id, 2 * FFTSizePerWI,
                                                         static_cast<std::size_t>(2 * M * row));

    detail::unrolled_loop<0, FFTSizePerWI, 1>([&](const int i) __attribute__((always_inline)) {
      int element = static_cast<int>(fft_local_id) * FFTSizePerWI + i;
      int twiddle_index = 2 * (M * row + element);
      T twiddle_real = wg_twiddles[twiddle_index];
      T twiddle_imag = wg_twiddles[twiddle_index + 1];
      if constexpr (Dir == direction::BACKWARD) {
        twiddle_imag = -twiddle_imag;
      }
      T tmp_real = priv[2 * i];
      priv[2 * i] = tmp_real * twiddle_real - priv[2 * i + 1] * twiddle_imag;
      priv[2 * i + 1] = tmp_real * twiddle_imag + priv[2 * i + 1] * twiddle_real;
    });

    sg_dft<Dir, FFTSizePerWI, WIPerFFT>(priv, sg, loc_twiddles);
    detail::unrolled_loop<0, FFTSizePerWI, 1>([&](const int i) __attribute__((always_inline)) {
      priv[2 * i] *= scaling_factor;
      priv[2 * i + 1] *= scaling_factor;
    });

    store_transposed<2 * FFTSizePerWI, detail::pad::DO_PAD>(priv, loc, fft_local_id, WIPerFFT,
                                                            static_cast<std::size_t>(2 * M * row));
  }
}

/**
 * Calculates FFT using Bailey 4 step algorithm.
 *
 * @tparam Dir Direction of the FFT
 * @tparam N Smaller factor of the Problem size
 * @tparam M Larger factor of the problem size
 * @tparam SubgroupSize Size of the subgroup
 * @tparam T Scalar Type
 *
 * @param loc local accessor containing the input
 * @param loc_twiddles Pointer to twiddles to be used by sub group FFTs
 * @param wg_twiddles Pointer to precalculated twiddles which are to be used before second set of FFTs
 * @param it Associated nd_item
 * @param scaling_factor Scalar value with which the result is to be scaled
 */
template <direction Dir, int N, int M, int SubgroupSize, typename T>
__attribute__((always_inline)) inline void wg_dft(T* loc, T* loc_twiddles, const T* wg_twiddles, sycl::nd_item<1> it,
                                                  T scaling_factor) {
  // factor_sg find the biggest factor <= sg_size
  constexpr int fact_sg_N = detail::factorize_sg(N, SubgroupSize);
  constexpr int fact_wi_N = N / fact_sg_N;

  constexpr int fact_sg_M = detail::factorize_sg(M, SubgroupSize);
  constexpr int fact_wi_M = M / fact_sg_M;

  if constexpr (N * M == 4096) {
    static_assert(N == 64);
    static_assert(M == 64);
    static_assert(fact_sg_N == 32);
    static_assert(fact_sg_M == 32);
    static_assert(fact_wi_N == 2);
    static_assert(fact_wi_M == 2);
  }

  constexpr int private_mem_size = 2 * (fact_wi_M > fact_wi_N ? fact_wi_M : fact_wi_N);
  T priv[private_mem_size];

  sycl::sub_group sg = it.get_sub_group();

  column_ffts<Dir, N, M, fact_sg_N, fact_wi_N, SubgroupSize>(sg, priv, loc, loc_twiddles);
  sycl::group_barrier(it.get_group());
  row_ffts<Dir, N, M, fact_sg_M, fact_wi_M, SubgroupSize>(sg, priv, loc, loc_twiddles, wg_twiddles, scaling_factor);
}

}  // namespace sycl_fft

#endif
