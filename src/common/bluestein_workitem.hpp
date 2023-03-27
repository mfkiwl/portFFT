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

#include <common/helpers.hpp>
#include <sycl/sycl.hpp>

// For small sizes, a naive approach is probably more than a Bluestein FFT.

#ifndef SYCL_FFT_COMMON_BLUESTEIN_WORKITEM_HPP
#define SYCL_FFT_COMMON_BLUESTEIN_WORKITEM_HPP

namespace sycl_fft::detail {

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