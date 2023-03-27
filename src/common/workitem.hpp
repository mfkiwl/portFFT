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

#ifndef SYCL_FFT_COMMON_WORKITEM_HPP
#define SYCL_FFT_COMMON_WORKITEM_HPP

#include <sycl/sycl.hpp>
#include <common/helpers.hpp>
#include <common/twiddle.hpp>

namespace sycl_fft{

//forward declaration
template <int N, int stride_in, int stride_out, typename T_ptr>
inline void wi_dft(T_ptr in, T_ptr out);

namespace detail{

/**
 * Calculates DFT using naive algorithm. Can work in or out of place.
 * 
 * @tparam N size of the DFT transform
 * @tparam stride_in stride (in complex values) between complex values in `in`
 * @tparam stride_out stride (in complex values) between complex values in `out`
 * @tparam T_ptr type of pointer for `in` and `out`. Can be raw pointer or sycl::multi_ptr.
 * @param in pointer to input
 * @param out pointer to output
*/
template <int N, int stride_in, int stride_out, typename T_ptr>
inline __attribute__((always_inline)) void naive_dft(T_ptr in, T_ptr out) {
    using T = remove_ptr<T_ptr>;
    constexpr T TWOPI = T(2.0) * M_PI;
    T tmp[2*N];
    unrolled_loop<0, N, 1>([&](int idx_out) __attribute__((always_inline)) {
      tmp[2 * idx_out + 0] = 0;
      tmp[2 * idx_out + 1] = 0;
      unrolled_loop<0, N, 1>([&](int idx_in) __attribute__((always_inline)) {
        // this multiplier is not really a twiddle factor, but it is calculated
        // the same way
        const T multi_re = twiddle<T>::re[N][idx_in * idx_out % N];
        const T multi_im = twiddle<T>::im[N][idx_in * idx_out % N];

        // multiply in and multi
        tmp[2 * idx_out + 0] += in[2 * idx_in * stride_in] * multi_re -
                                in[2 * idx_in * stride_in + 1] * multi_im;
        tmp[2 * idx_out + 1] += in[2 * idx_in * stride_in] * multi_im +
                                in[2 * idx_in * stride_in + 1] * multi_re;
      });
    });
    unrolled_loop<0, 2 * N, 2>([&](int idx_out) {
      out[idx_out * stride_out + 0] = tmp[idx_out + 0];
      out[idx_out * stride_out + 1] = tmp[idx_out + 1];
    });
}

//mem requirement: ~N*M(if in place, otherwise x2) + N*M(=tmp) + sqrt(N*M) + pow(N*M,0.25) + ...
// TODO explore if this tmp can be reduced/eliminated ^^^^^^
/**
 * Calculates DFT using Cooley-Tukey FFT algorithm. Can work in or out of place. Size of the problem is N*M
 * 
 * @tparam N the first factor of the problem size
 * @tparam M the second factor of the problem size
 * @tparam stride_in stride (in complex values) between complex values in `in`
 * @tparam stride_out stride (in complex values) between complex values in `out`
 * @tparam T_ptr type of pointer for `in` and `out`. Can be raw pointer or sycl::multi_ptr.
 * @param in pointer to input
 * @param out pointer to output
*/
template <int N, int M, int stride_in, int stride_out, typename T_ptr>
inline __attribute__((always_inline)) void cooley_tukey_dft(T_ptr in, T_ptr out) {
    using T = remove_ptr<T_ptr>;
    T tmp_buffer[2*N*M];
    unrolled_loop<0,M,1>([&](int i) __attribute__((always_inline)) {
        wi_dft<N, M*stride_in, 1>(in + 2*i*stride_in, tmp_buffer + 2*i*N);
        unrolled_loop<0,N,1>([&](int j) __attribute__((always_inline)) {
            T tmp_val = tmp_buffer[2*i*N + 2*j] * twiddle<T>::re[N*M][i*j] - tmp_buffer[2*i*N + 2*j + 1] * twiddle<T>::im[N*M][i*j];
            tmp_buffer[2*i*N + 2*j + 1] = tmp_buffer[2*i*N + 2*j] * twiddle<T>::im[N*M][i*j] + tmp_buffer[2*i*N + 2*j + 1] * twiddle<T>::re[N*M][i*j];
            tmp_buffer[2*i*N + 2*j + 0] = tmp_val;
        });
    });
    unrolled_loop<0,N,1>([&](int i) __attribute__((always_inline)) {
        wi_dft<M, N, N*stride_out>(tmp_buffer + 2*i, out + 2*i*stride_out);
    });
}

/**
 * Factorizes a number into two roughly equal factors.
 * @param N the number to factorize
 * @return the smaller of the factors
 */
constexpr int factorize(int N) {
  int res = 1;
  for (int i = 2; i * i <= N; i++) {
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
 * @return Number of temporary complex values
 */
constexpr int wi_temps(int N) {
  int F0 = factorize(N);
  int F1 = N / F0;
  if (F0 < 2 || F1 < 2) {
    return N;
  }
  int a = wi_temps(F0);
  int b = wi_temps(F1);
  return (a > b ? a : b) + N;
}

/**
 * Checks whether a problem can be solved with workitem implementation without
 * registers spilling.
 * @tparam Scalar type of the real scalar used for the computation
 * @param N Size of the problem, in complex values
 * @return true if the problem fits in the registers
 */
template <typename Scalar>
constexpr bool fits_in_wi(int N) {
  int N_complex = N + wi_temps(N);
  int complex_size = 2 * sizeof(Scalar);
  int register_space = SYCLFFT_TARGET_REGS_PER_WI * 4;
  return N_complex * complex_size <= register_space;
}

/**
 * Struct with precalculated values for all relevant arguments to
 * fits_in_wi for use on device, where recursive functions are not allowed.
 *
 * @tparam Scalar type of the real scalar used for the computation
 */
template <typename Scalar>
struct fits_in_wi_device_struct {
  static constexpr bool buf[56] = {
      fits_in_wi<Scalar>(1),  fits_in_wi<Scalar>(2),  fits_in_wi<Scalar>(3),
      fits_in_wi<Scalar>(4),  fits_in_wi<Scalar>(5),  fits_in_wi<Scalar>(6),
      fits_in_wi<Scalar>(7),  fits_in_wi<Scalar>(8),  fits_in_wi<Scalar>(9),
      fits_in_wi<Scalar>(10), fits_in_wi<Scalar>(11), fits_in_wi<Scalar>(12),
      fits_in_wi<Scalar>(13), fits_in_wi<Scalar>(14), fits_in_wi<Scalar>(15),
      fits_in_wi<Scalar>(16), fits_in_wi<Scalar>(17), fits_in_wi<Scalar>(18),
      fits_in_wi<Scalar>(19), fits_in_wi<Scalar>(20), fits_in_wi<Scalar>(21),
      fits_in_wi<Scalar>(22), fits_in_wi<Scalar>(23), fits_in_wi<Scalar>(24),
      fits_in_wi<Scalar>(25), fits_in_wi<Scalar>(26), fits_in_wi<Scalar>(27),
      fits_in_wi<Scalar>(28), fits_in_wi<Scalar>(29), fits_in_wi<Scalar>(30),
      fits_in_wi<Scalar>(31), fits_in_wi<Scalar>(32), fits_in_wi<Scalar>(33),
      fits_in_wi<Scalar>(34), fits_in_wi<Scalar>(35), fits_in_wi<Scalar>(36),
      fits_in_wi<Scalar>(37), fits_in_wi<Scalar>(38), fits_in_wi<Scalar>(39),
      fits_in_wi<Scalar>(40), fits_in_wi<Scalar>(41), fits_in_wi<Scalar>(42),
      fits_in_wi<Scalar>(43), fits_in_wi<Scalar>(44), fits_in_wi<Scalar>(45),
      fits_in_wi<Scalar>(46), fits_in_wi<Scalar>(47), fits_in_wi<Scalar>(48),
      fits_in_wi<Scalar>(49), fits_in_wi<Scalar>(50), fits_in_wi<Scalar>(51),
      fits_in_wi<Scalar>(52), fits_in_wi<Scalar>(53), fits_in_wi<Scalar>(54),
      fits_in_wi<Scalar>(55), fits_in_wi<Scalar>(56),
  };
};

/**
 * Checks whether a problem can be solved with workitem implementation without
 * registers spilling. Non-recursive implementation for the use on device.
 * @tparam Scalar type of the real scalar used for the computation
 * @param N Size of the problem, in complex values
 * @return true if the problem fits in the registers
 */
template <typename Scalar>
bool fits_in_wi_device(int fft_size) {
  // 56 is the maximal size we support in workitem implementation and also
  // the size of the array above that is used if this if is not taken
  if (fft_size > 56) {
    return false;
  }
  return fits_in_wi_device_struct<Scalar>::buf[fft_size - 1];
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

};  // namespace detail

/**
 * Calculates DFT using FFT algorithm. Can work in or out of place.
 *
 * @tparam N size of the DFT transform
 * @tparam stride_in stride (in complex values) between complex values in `in`
 * @tparam stride_out stride (in complex values) between complex values in `out`
 * @tparam T_ptr type of pointer for `in` and `out`. Can be raw pointer or
 * sycl::multi_ptr.
 * @param in pointer to input
 * @param out pointer to output
 */
template <int N, int stride_in, int stride_out, typename T_ptr>
inline __attribute__((always_inline)) void wi_dft(T_ptr in, T_ptr out) {
  constexpr int F0 = detail::factorize(N);
  if constexpr (N == 2) {
    using T = detail::remove_ptr<T_ptr>;
    T a = in[0 * stride_in + 0] + in[2 * stride_in + 0];
    T b = in[0 * stride_in + 1] + in[2 * stride_in + 1];
    T c = in[0 * stride_in + 0] - in[2 * stride_in + 0];
    out[2 * stride_out + 1] = in[0 * stride_in + 1] - in[2 * stride_in + 1];
    out[0 * stride_out + 0] = a;
    out[0 * stride_out + 1] = b;
    out[2 * stride_out + 0] = c;
  } else if constexpr (F0 >= 2 && N / F0 >= 2) {
    detail::cooley_tukey_dft<N / F0, F0, stride_in, stride_out>(in, out);
  } else {
    detail::naive_dft<N, stride_in, stride_out>(in, out);
  }
}

}; //namespace sycl_fft

#endif
