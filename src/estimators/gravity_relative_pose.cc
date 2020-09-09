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

#include "estimators/gravity_relative_pose.h"

#include <Eigen/Eigenvalues>

#include "base/pose.h"
#include "estimators/utils.h"

namespace colmap {

void GravityRelativePoseEstimator::SetRotationAxis(
    const Eigen::Vector3d& rotation_axis) {
  rotation_axis_ = rotation_axis.normalized();
}

std::vector<Pose> GravityRelativePoseEstimator::Estimate(
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2) const {
  // Assemble the QEP, see equation 5 and 6 in Section 3.1 of
  // Sweeney et al. 2014

  const Eigen::Vector3d& v = rotation_axis_;

  Eigen::Matrix3d M;  // quadratic part
  Eigen::Matrix3d C;  // linear part
  Eigen::Matrix3d K;  // constant part

  for (int i = 0; i < 3; ++i) {
    const Eigen::Vector3d p1 = points1[i].homogeneous();
    const Eigen::Vector3d p2 = points2[i].homogeneous();

    M.row(i) = p2.cross(p1);
    C.row(i) = 2. * p2.cross(v.cross(p1));
    K.row(i) = 2. * v.dot(p1) * p2.cross(v) - p2.cross(p1);
  }

  // Solve QEP

  std::vector<Pose> candidate_solutions;

  Eigen::Matrix3d M_inv;
  bool M_invertible;
  M.computeInverseWithCheck(M_inv, M_invertible, 1e-12);

  if (M_invertible) {
    Eigen::Matrix<double, 6, 6> SEP;
    SEP <<
        -M_inv * C,                  -M_inv * K,
        Eigen::Matrix3d::Identity(), Eigen::Matrix3d::Zero();
    Eigen::EigenSolver<Eigen::Matrix<double, 6, 6>> eigensolver(SEP);

    candidate_solutions.reserve(4);
    for (int i = 0; i < 6; ++i) {
      if (std::abs(eigensolver.eigenvalues()[i].imag()) < 0.001) {
        const double s = eigensolver.eigenvalues()[i].real();

        Eigen::Vector4d rotation(s, v[0], v[1], v[2]);
        rotation.normalize();

        const Eigen::Vector3d translation =
            eigensolver.eigenvectors().col(i).tail<3>().real().normalized();

        candidate_solutions.push_back({rotation, translation});
      }
    }

    // assert(candidate_solutions.size() >= 4);
  } else {
    candidate_solutions.reserve(2);

    const Eigen::Vector4d zero_rot = ComposeIdentityQuaternion();

    Eigen::JacobiSVD<Eigen::Matrix3d> svd = M.jacobiSvd(Eigen::ComputeFullV);
    const Eigen::Vector3d nullspace_M = svd.matrixV().col(2).normalized();

    candidate_solutions.push_back({zero_rot, nullspace_M});
    candidate_solutions.push_back({zero_rot, -nullspace_M});
  }

  return candidate_solutions;
}

void GravityRelativePoseEstimator::Residuals(
    const std::vector<Eigen::Vector2d>& points1,
    const std::vector<Eigen::Vector2d>& points2, const Pose& transformation,
    std::vector<double>* residuals) {
  const Eigen::Matrix3d E = CrossProductMatrix(transformation.tvec) *
                            QuaternionToRotationMatrix(transformation.qvec);
  ComputeSquaredSampsonError(points1, points2, E, residuals);
}

}  // namespace colmap
