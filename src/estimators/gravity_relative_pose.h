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

#ifndef COLMAP_SRC_ESTIMATORS_GRAVITY_RELATIVE_POSE_H_
#define COLMAP_SRC_ESTIMATORS_GRAVITY_RELATIVE_POSE_H_

#include <vector>

#include <Eigen/Core>

#include "base/pose.h"

namespace colmap {

// Relative Pose from two 2D-2D correspondences with a known axis of rotation.
//
// The algorithm is based on the following paper:
//
//    C. Sweeney, J. Flynn, M. Turk. Solving for Relative Pose with a Partially
//    Known Rotation is a Quadratic Eigenvalue Problem.

class GravityRelativePoseEstimator {
 public:
  // The 2D points in image 1.
  using X_t = Eigen::Vector2d;
  // The 2D points in image 2.
  using Y_t = Eigen::Vector2d;
  // The transformation from the coordinate system of the first camera
  // to the second camera.
  using M_t = Pose;

  // The minimum number of samples needed to estimate a model.
  static const int kMinNumSamples = 3;

  void SetRotationAxis(const Eigen::Vector3d& rotation_axis);

  // Estimate the most probable pose based on the given correspondences and the
  // rotation axis in the state of the GravityRelativePoseEstimator object.
  //
  // @param points1   Normalized 2D points in first image as 2D vectors.
  // @param points2   Normalized 2D points in second image as 2D vectors.
  std::vector<M_t> Estimate(const std::vector<X_t>& points1,
                            const std::vector<Y_t>& points2) const;

  // Calculate the squared reprojection error in normalized coordinates.
  //
  // @param points1         Normalized 2D points in first image as 2D vectors.
  // @param points2         Normalized 2D points in second image as 2D vectors.
  // @param transformation  Transformation from second to first image coordinate
  // frame.
  // @param residuals       Output vector of residuals.
  static void Residuals(const std::vector<X_t>& points1,
                        const std::vector<Y_t>& points2,
                        const M_t& transformation,
                        std::vector<double>* residuals);

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

 private:
  Eigen::Vector3d rotation_axis_;
};

}  // namespace colmap

#endif  // COLMAP_SRC_ESTIMATORS_GRAVITY_RELATIVE_POSE_H_
