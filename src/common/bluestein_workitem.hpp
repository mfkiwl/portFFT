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

#include <sycl/sycl.hpp>
#include <common/helpers.hpp>

// For small sizes, a naive approach is probably more than a Bluestein FFT.

#ifndef SYCL_FFT_COMMON_BLUESTEIN_WORKITEM_HPP
#define SYCL_FFT_COMMON_BLUESTEIN_WORKITEM_HPP

namespace sycl_fft::detail {
template <typename T>
inline __attribute__((always_inline)) void complex_mult(T* x, T* y, T* out) {
  T a = x[0];
  T b = x[1];
  T c = y[0];
  T d = y[1];
  out[0] = a * c - b * d;
  out[1] = b * c + a * d;
}

// TODO replace with a fftconvolution when inverse dft is merged
template <int N, typename T>
inline __attribute__((always_inline)) void wi_convolution(T* f, T* g, T* out) {
  unrolled_loop<0, 2 * N, 2>([&](int n) __attribute__((always_inline)) {
    out[n] = T(0.0);
    out[n + 1] = T(0.0);
    int m = 0;
    for (; m != n + 2; m += 2) {
      T mult[2];
      complex_mult(f + m, g + (n - m), mult);
      out[n] += mult[0];
      out[n + 1] += mult[1];
    }
    for (; m != 2 * N; m += 2) {
      T mult[2];
      complex_mult(f + m, g + (2 * N + n - m), mult);
      out[n] += mult[0];
      out[n + 1] += mult[1];
    }
  });
}

template <int N, int N_padded, typename T>
inline __attribute__((always_inline)) void wi_bluestein_dft(T* in, T* out, T* g,
                                                            T* multiplier) {
  // 2*N-1 is probably a bad idea performance wise, but it's the mathematical
  // minimum
  static_assert(N_padded >= 2 * N - 1);
  static_assert(std::is_same<T, float>::value, "expected an array of floats");

  T f[2 * N_padded] = {0};
  // g = e^((n^2 * pi * i)/N)
  // f = in[n]*e^(-(n^2 * pi * i)/N)
  unrolled_loop<0, 2 * N, 2>([&](int n) {
    // this works too!
    // T f_partial[2];
    // f_partial[0] = g[n];
    // f_partial[1] = -g[n+1];
    // complex_mult(f_partial, in+n, f+n);
    complex_mult(multiplier + n, in + n, f + n);
  });

  // TODO in the future f and g will be convolved with an FFT convolution, i.e.
  // the inverse FFT of the produce of two FFTs. At that point we can actually
  // pre-calculate the FFT of g. Because of this g will no longer be present so
  // the multiplier array will be needed.

  // convolve g and f
  T conv_out[2 * N_padded];
  wi_convolution<N_padded, T>(f, g, conv_out);

  // multiply by constant, W^(-(k^2)/2)
  // = e^(i(-pi k^2)/N) = cos((-pi k^2)/N) + isin((-pi k^2)/N)
  unrolled_loop<0, 2 * N, 2>([&](int n) {
    // this works too!
    // T m[2];
    // m[0] = g[n];
    // m[1] = -g[n+1];
    // complex_mult(m, conv_out + n, out + n);
    complex_mult(multiplier + n, conv_out + n, out + n);
  });
}
}  // namespace sycl_fft::detail

#endif