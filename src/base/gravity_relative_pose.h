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

#ifndef COLMAP_SRC_BASE_GRAVITY_RELATIVE_POSE_H_
#define COLMAP_SRC_BASE_GRAVITY_RELATIVE_POSE_H_

#include <array>
#include <vector>

#include <Eigen/Core>

#include "base/camera.h"
#include "base/image.h"
#include "base/pose.h"
#include "estimators/gravity_relative_pose.h"
#include "feature/types.h"
#include "optim/loransac.h"

namespace colmap {

// Rotation to align the image vertical axis with the global vertical axis.
inline Eigen::Quaterniond AxisAlignmentFromImage(const Image& image) {
  return Eigen::Quaterniond::FromTwoVectors(image.GravityPrior(),
                                            Eigen::Vector3d(0, 1, 0));
}

LORANSAC<GravityRelativePoseEstimator, GravityRelativePoseEstimator>::Report
EstimateRelativePoseGravity(
    std::array<std::vector<Eigen::Vector2d>, 2> normalized_keypoints,
    const std::array<Eigen::Vector3d, 2>& gravity,
    const RANSACOptions ransac_options, std::array<Pose, 2>* poses);

LORANSAC<GravityRelativePoseEstimator, GravityRelativePoseEstimator>::Report
EstimateRelativePoseGravity(const std::array<Image, 2>& images,
                            const std::array<Camera, 2>& cameras,
                            const FeatureMatches matches,
                            const RANSACOptions ransac_options,
                            std::array<Pose, 2>* poses);

}  // namespace colmap

#endif  // COLMAP_SRC_BASE_GRAVITY_RELATIVE_POSE_H_
