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

#ifndef PORTFFT_COMMON_WORKITEM_HPP
#define PORTFFT_COMMON_WORKITEM_HPP

#include <common/helpers.hpp>
#include <common/twiddle.hpp>
#include <enums.hpp>
#include <sycl/sycl.hpp>

namespace portfft {

// forward declaration
template <direction Dir, int N, int StrideIn, int StrideOut, typename T>
inline void wi_dft(const T* in, T* out);

namespace detail {

// Maximum size of an FFT that can fit in the workitem implementation
static constexpr std::size_t MaxFftSizeWi = 56;

/*
`wi_dft` calculates a DFT by a workitem on values that are already loaded into its private memory.
It calls either `cooley_tukey_dft` (for composite sizes) or `naive_dft` (for prime sizes).

`cooley_tukey_dft` calculates DFT of a composite size by one workitem. It calls `wi_dft` for each of the factors and
does twiddle multiplication in-between. Transposition is handled by calling `wi_dft` with different input and output
strides.

`naive_dft` calculates DFT by one workitem using naive DFT algorithm.
*/

/**
 * Calculates DFT using naive algorithm. Can work in or out of place.
 *
 * @tparam Dir direction of the FFT
 * @tparam N size of the DFT transform
 * @tparam StrideIn stride (in complex values) between complex values in `in`
 * @tparam StrideOut stride (in complex values) between complex values in `out`
 * @tparam T type of the scalar used for computations
 * @param in pointer to input
 * @param out pointer to output
 */
template <direction Dir, int N, int StrideIn, int StrideOut, typename T>
__attribute__((always_inline)) inline void naive_dft(const T* in, T* out) {
  T tmp[2 * N];
#pragma unroll
  for (int idx_out = 0; idx_out < N; idx_out += 1) {
    tmp[2 * idx_out + 0] = 0;
    tmp[2 * idx_out + 1] = 0;
#pragma unroll
    for (int idx_in = 0; idx_in < N; idx_in += 1) {
      // this multiplier is not really a twiddle factor, but it is calculated the same way
      auto re_multiplier = twiddle<T>::Re[N][idx_in * idx_out % N];
      auto im_multiplier = [&]() {
        if constexpr (Dir == direction::FORWARD) {
          return twiddle<T>::Im[N][idx_in * idx_out % N];
        }
        return -twiddle<T>::Im[N][idx_in * idx_out % N];
      }();

      // multiply in and multi
      tmp[2 * idx_out + 0] += in[2 * idx_in * StrideIn] * re_multiplier - in[2 * idx_in * StrideIn + 1] * im_multiplier;
      tmp[2 * idx_out + 1] += in[2 * idx_in * StrideIn] * im_multiplier + in[2 * idx_in * StrideIn + 1] * re_multiplier;
    }
  }
#pragma unroll
  for (int idx_out = 0; idx_out < 2 * N; idx_out += 2) {
    out[idx_out * StrideOut + 0] = tmp[idx_out + 0];
    out[idx_out * StrideOut + 1] = tmp[idx_out + 1];
  }
}

// mem requirement: ~N*M(if in place, otherwise x2) + N*M(=tmp) + sqrt(N*M) + pow(N*M,0.25) + ...
// TODO explore if this tmp can be reduced/eliminated ^^^^^^
/**
 * Calculates DFT using Cooley-Tukey FFT algorithm. Can work in or out of place. Size of the problem is N*M
 *
 * @tparam Dir direction of the FFT
 * @tparam N the first factor of the problem size
 * @tparam M the second factor of the problem size
 * @tparam StrideIn stride (in complex values) between complex values in `in`
 * @tparam StrideOut stride (in complex values) between complex values in `out`
 * @tparam T type of the scalar used for computations
 * @param in pointer to input
 * @param out pointer to output
 */
template <direction Dir, int N, int M, int StrideIn, int StrideOut, typename T>
__attribute__((always_inline)) inline void cooley_tukey_dft(const T* in, T* out) {
  T tmp_buffer[2 * N * M];

#pragma unroll
  for (int i = 0; i < M; i += 1) {
    wi_dft<Dir, N, M * StrideIn, 1>(in + 2 * i * StrideIn, tmp_buffer + 2 * i * N);
#pragma unroll
    for (int j = 0; j < N; j += 1) {
      auto re_multiplier = twiddle<T>::Re[N * M][i * j];
      auto im_multiplier = [&]() {
        if constexpr (Dir == direction::FORWARD) {
          return twiddle<T>::Im[N * M][i * j];
        }
        return -twiddle<T>::Im[N * M][i * j];
      }();
      T tmp_val = tmp_buffer[2 * i * N + 2 * j] * re_multiplier - tmp_buffer[2 * i * N + 2 * j + 1] * im_multiplier;
      tmp_buffer[2 * i * N + 2 * j + 1] =
          tmp_buffer[2 * i * N + 2 * j] * im_multiplier + tmp_buffer[2 * i * N + 2 * j + 1] * re_multiplier;
      tmp_buffer[2 * i * N + 2 * j + 0] = tmp_val;
    }
  }
#pragma unroll
  for (int i = 0; i < N; i += 1) {
    wi_dft<Dir, M, N, N * StrideOut>(tmp_buffer + 2 * i, out + 2 * i * StrideOut);
  }
}

/**
 * Factorizes a number into two roughly equal factors.
 * @tparam TIndex Index type
 * @param N the number to factorize
 * @return the smaller of the factors
 */
template <typename TIndex>
constexpr TIndex factorize(TIndex N) {
  TIndex res = 1;
  for (TIndex i = 2; i * i <= N; i++) {
    if (N % i == 0) {
      res = i;
    }
  }
  return res;
}

/**
 * Calculates how many temporary complex values a workitem implementation needs
 * for solving FFT.
 * @param N size of the FFT problem
 * @tparam TIndex Index type
 * @return Number of temporary complex values
 */
template <typename TIndex>
constexpr TIndex wi_temps(TIndex N) {
  TIndex f0 = factorize(N);
  TIndex f1 = N / f0;
  if (f0 < 2 || f1 < 2) {
    return N;
  }
  TIndex a = wi_temps(f0);
  TIndex b = wi_temps(f1);
  return (a > b ? a : b) + N;
}

/**
 * Checks whether a problem can be solved with workitem implementation without
 * registers spilling.
 * @tparam Scalar type of the real scalar used for the computation
 * @tparam TIndex Index type
 * @param N Size of the problem, in complex values
 * @return true if the problem fits in the registers
 */
template <typename Scalar, typename TIndex>
constexpr bool fits_in_wi(TIndex N) {
  TIndex n_complex = N + wi_temps(N);
  TIndex complex_size = 2 * sizeof(Scalar);
  TIndex register_space = PORTFFT_REGISTERS_PER_WI * 4;
  return n_complex * complex_size <= register_space;
}

};  // namespace detail

/**
 * Calculates DFT using FFT algorithm. Can work in or out of place.
 *
 * @tparam Dir direction of the FFT
 * @tparam N size of the DFT transform
 * @tparam StrideIn stride (in complex values) between complex values in `in`
 * @tparam StrideOut stride (in complex values) between complex values in `out`
 * @tparam T type of the scalar used for computations
 * @param in pointer to input
 * @param out pointer to output
 */
template <direction Dir, int N, int StrideIn, int StrideOut, typename T>
__attribute__((always_inline)) inline void wi_dft(const T* in, T* out) {
  constexpr int F0 = detail::factorize(N);
  if constexpr (N == 2) {
    T a = in[0 * StrideIn + 0] + in[2 * StrideIn + 0];
    T b = in[0 * StrideIn + 1] + in[2 * StrideIn + 1];
    T c = in[0 * StrideIn + 0] - in[2 * StrideIn + 0];
    out[2 * StrideOut + 1] = in[0 * StrideIn + 1] - in[2 * StrideIn + 1];
    out[0 * StrideOut + 0] = a;
    out[0 * StrideOut + 1] = b;
    out[2 * StrideOut + 0] = c;
  } else if constexpr (F0 >= 2 && N / F0 >= 2) {
    detail::cooley_tukey_dft<Dir, N / F0, F0, StrideIn, StrideOut>(in, out);
  } else {
    detail::naive_dft<Dir, N, StrideIn, StrideOut>(in, out);
  }
}

};  // namespace portfft

#endif
