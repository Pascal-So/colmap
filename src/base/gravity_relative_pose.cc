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

#include "base/gravity_relative_pose.h"

#include "estimators/gravity_relative_pose.h"
#include "optim/loransac.h"

namespace colmap {

unsigned EstimateRelativePoseGravity(const std::array<Image, 2>& images,
                                     const std::array<Camera, 2>& cameras,
                                     const FeatureMatches matches,
                                     const RANSACOptions ransac_options,
                                     std::array<Pose, 2>* poses) {
  const std::array<Eigen::Quaterniond, 2> rotations{
      AxisAlignmentFromImage(images[0]), AxisAlignmentFromImage(images[1])};

  std::array<std::vector<Eigen::Vector2d>, 2> normalized_rotated_keypoints;
  normalized_rotated_keypoints[0].reserve(matches.size());
  normalized_rotated_keypoints[1].reserve(matches.size());

  for (std::size_t match_id = 0; match_id < matches.size(); ++match_id) {
    const std::array<point2D_t, 2> idx{matches[match_id].point2D_idx1,
                                       matches[match_id].point2D_idx2};

    for (int i : {0, 1}) {
      const Eigen::Vector2d normalized =
          cameras[i].ImageToWorld(images[i].Points2D()[idx[i]].XY());

      const Eigen::Vector2d normalized_rotated =
          (rotations[i] * normalized.homogeneous()).hnormalized();

      normalized_rotated_keypoints[i].push_back(normalized_rotated);
    }
  }

  LORANSAC<GravityRelativePoseEstimator, GravityRelativePoseEstimator> ransac(
      ransac_options);

  const auto ransac_report = ransac.Estimate(normalized_rotated_keypoints[0],
                                             normalized_rotated_keypoints[1]);

  if (!ransac_report.success) {
    return 0;
  }

  // TODO: compute and assign poses

  return ransac_report.support.num_inliers;
}

}  // namespace colmap
