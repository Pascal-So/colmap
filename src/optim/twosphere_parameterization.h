// Copyright (c) 2020, ETH Zurich
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: Pascal Sommer (pasommer-at-student-dot-ethz-dot-ch)

#ifndef COLMAP_SRC_OPTIM_TWOSPHERE_PARAMETERIZATION_H_
#define COLMAP_SRC_OPTIM_TWOSPHERE_PARAMETERIZATION_H_

#include <cassert>
#include <cmath>

#include <Eigen/Core>

#include <ceres/autodiff_local_parameterization.h>

namespace colmap {

// The jacobian is discontinuous in x but I guess ceres won't care since
// I don't think it will keep information across evaluations.

// The 2-sphere doesn't neccessarily have to be of unit size. We only
// make sure that the norm remains constant, not that the norm is equal
// to 1.

struct TwospherePlus {
  template<typename T>
  bool operator()(const T* x, const T* delta, T* x_plus_delta) const {
    using namespace Eigen;

    using vec3_t = Matrix<T, 3, 1>;
    using vec2_t = Matrix<T, 2, 1>;

    const Map<const vec3_t> v_x(x);
    const Map<const vec2_t> v_delta(delta);
    Map<vec3_t> v_x_plus_delta(x_plus_delta);

    T norm_x = v_x.norm();

    vec3_t a(T(1), T(0), T(0));
    vec3_t b = v_x.cross(a);

    if (b.squaredNorm() < 1e-8) {
      a << T(0), T(1), T(0);
      b = v_x.cross(a);
    }

    b.normalize();
    a = v_x.cross(a) / norm_x;

    v_x_plus_delta = v_x + a * v_delta(0) + b * v_delta(1);
    v_x_plus_delta *= norm_x / v_x_plus_delta.norm();
    return true;
  }
};

using TwosphereParameterization =
    ceres::AutoDiffLocalParameterization<TwospherePlus, 3, 2>;

}  // namespace colmap

#endif  // COLMAP_SRC_OPTIM_TWOSPHERE_PARAMETERIZATION_H_
