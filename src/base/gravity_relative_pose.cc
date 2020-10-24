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

#include <utility>

#include "base/pose.h"

namespace colmap {

LORANSAC<GravityRelativePoseEstimator, GravityRelativePoseEstimator>::Report
EstimateRelativePoseGravity(
    std::array<std::vector<Eigen::Vector2d>, 2> normalized_keypoints,
    const std::array<Eigen::Vector3d, 2>& gravity,
    const RANSACOptions ransac_options, std::array<Pose, 2>* poses,
    std::vector<Eigen::Vector3d>* points3D) {
  CHECK_NOTNULL(poses);

  std::array<Eigen::Quaterniond, 2> rotations;

  for (int i : {0, 1}) {
    rotations[i] = Eigen::Quaterniond::FromTwoVectors(gravity[i],
                                                      Eigen::Vector3d(0, 1, 0));

    for (auto& v : normalized_keypoints[i]) {
      // Todo: figure out what to do if the point lies on the xy plane after the
      // rotation
      v = (rotations[i] * v.homogeneous()).hnormalized();
    }
  }

  LORANSAC<GravityRelativePoseEstimator, GravityRelativePoseEstimator> ransac(
      ransac_options);
  ransac.estimator.SetRotationAxis(Eigen::Vector3d::UnitY());

  const auto ransac_report =
      ransac.Estimate(normalized_keypoints[0], normalized_keypoints[1]);
  if (!ransac_report.success) {
    return ransac_report;
  }
  const std::size_t nr_inliers = ransac_report.support.num_inliers;

  const std::size_t nr_points = normalized_keypoints[0].size();
  std::array<std::vector<Eigen::Vector2d>, 2> inlier_points;
  for (auto i : {0, 1}) {
    inlier_points[i].reserve(nr_inliers);
    for (std::size_t point_id = 0; point_id < nr_points; ++point_id) {
      if (ransac_report.inlier_mask[point_id]) {
        inlier_points[i].push_back(normalized_keypoints[i][point_id]);
      }
    }
  }

  // Check the sign that the translation vector should have.
  const Eigen::Matrix3d R =
      QuaternionToRotationMatrix(ransac_report.model.qvec);
  Eigen::Vector3d t = ransac_report.model.tvec;

  std::vector<Eigen::Vector3d> points3D_in_front;
  points3D_in_front.reserve(nr_inliers);
  CheckCheirality(R, t, inlier_points[0], inlier_points[1], &points3D_in_front);
  if (points3D != nullptr) {
    *points3D = points3D_in_front;
  }

  const std::size_t score_positive = points3D_in_front.size();

  CheckCheirality(R, -t, inlier_points[0], inlier_points[1],
                  &points3D_in_front);

  if (points3D_in_front.size() > score_positive) {
    t *= -1;
    if (points3D != nullptr) {
      *points3D = points3D_in_front;
    }
  }

  (*poses)[0].qvec = EigenQuaternionToQuaternion(rotations[0].conjugate());
  (*poses)[0].tvec = Eigen::Vector3d::Zero();
  (*poses)[1].qvec = EigenQuaternionToQuaternion(
      rotations[1].conjugate() *
      QuaternionToEigenQuaternion(ransac_report.model.qvec));
  (*poses)[1].tvec = rotations[1].conjugate() * t;

  return ransac_report;
}

LORANSAC<GravityRelativePoseEstimator, GravityRelativePoseEstimator>::Report
EstimateRelativePoseGravity(const std::array<Image, 2>& images,
                            const std::array<Camera, 2>& cameras,
                            const FeatureMatches matches,
                            const RANSACOptions ransac_options,
                            std::array<Pose, 2>* poses,
                            std::vector<Eigen::Vector3d>* points3D) {
  CHECK_NOTNULL(poses);
  CHECK(images[0].HasGravityPrior());
  CHECK(images[1].HasGravityPrior());

  const std::size_t nr_points = matches.size();

  std::array<Eigen::Vector3d, 2> gravity = {images[0].GravityPrior(),
                                            images[1].GravityPrior()};

  std::array<std::vector<Eigen::Vector2d>, 2> normalized_keypoints;
  normalized_keypoints[0].reserve(nr_points);
  normalized_keypoints[1].reserve(nr_points);

  for (std::size_t match_id = 0; match_id < nr_points; ++match_id) {
    const std::array<point2D_t, 2> idx{matches[match_id].point2D_idx1,
                                       matches[match_id].point2D_idx2};

    for (int i : {0, 1}) {
      const Eigen::Vector2d normalized =
          cameras[i].ImageToWorld(images[i].Points2D()[idx[i]].XY());
      normalized_keypoints[i].push_back(normalized);
    }
  }

  return EstimateRelativePoseGravity(std::move(normalized_keypoints), gravity,
                                     ransac_options, poses, points3D);
}

}  // namespace colmap
