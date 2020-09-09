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

#define TEST_NAME "base/gravity_relative_pose"
#include "util/testing.h"

#include <Eigen/Geometry>

#include "base/gravity_relative_pose.h"
#include "base/projection.h"
#include "util/random.h"

using namespace colmap;
using namespace Eigen;

BOOST_AUTO_TEST_CASE(TestEstimateRelativePoseGravity) {
  SetPRNGSeed(0);

  const std::size_t kNumPoints = 100;

  std::vector<Eigen::Vector3d> points3D;
  for (std::size_t i = 0; i < kNumPoints; ++i) {
    points3D.emplace_back(RandomReal<double>(0, 1),
                          RandomReal<double>(-1, 1),
                          RandomReal<double>(0.5, 5));
  }

  std::array<Pose, 2> poses_orig;
  poses_orig[0].tvec = Vector3d::Zero();
  poses_orig[0].qvec = Vector4d(0.981445, 0.104679, -0.15143, 0.053634);
  poses_orig[1].tvec = Vector3d(-4, -0.5, 5).normalized();
  poses_orig[1].qvec = Vector4d(0.960427, -0.101919, 0.258302, -0.021735);

  std::array<Image, 2> images;
  std::array<Camera, 2> cameras;
  std::array<std::vector<Vector2d>, 2> projected;
  std::array<Matrix3x4d, 2> projection_matrix;
  for (auto i : {0, 1}) {
    images[i].SetGravityPrior(
        QuaternionRotatePoint(poses_orig[i].qvec, Vector3d::UnitY()));

    cameras[i].SetCameraId(i);
    cameras[i].InitializeWithName("PINHOLE", 1, 1, 1);

    projection_matrix[i] =
        ComposeProjectionMatrix(poses_orig[i].qvec, poses_orig[i].tvec);

    projected[i].reserve(kNumPoints);
  }

  FeatureMatches matches;
  matches.reserve(kNumPoints);

  for (std::size_t point_id = 0; point_id < kNumPoints; ++point_id) {
    matches.emplace_back(point_id, point_id);

    for (auto i : {0, 1}) {
      const Vector2d projected_normalized =
          (projection_matrix[i] * points3D[point_id].homogeneous())
              .hnormalized();
      projected[i].push_back(cameras[i].WorldToImage(projected_normalized));
    }
  }

  for (auto i : {0, 1}) {
    images[i].SetPoints2D(projected[i]);
  }

  RANSACOptions ransac_options;
  ransac_options.min_num_trials = 1;
  ransac_options.max_num_trials = 1;
  ransac_options.max_error = 1e-5;

  std::array<Pose, 2> poses_est;

  unsigned nr_inliers = EstimateRelativePoseGravity(images, cameras, matches,
                                                    ransac_options, &poses_est);

  BOOST_CHECK_EQUAL(poses_est[0].tvec, Vector3d::Zero());
  BOOST_CHECK_CLOSE(poses_est[0].qvec.norm(), 1., 1e-9);
  BOOST_CHECK_CLOSE(poses_est[1].tvec.norm(), 1., 1e-9);
  BOOST_CHECK_CLOSE(poses_est[1].qvec.norm(), 1., 1e-9);
  BOOST_CHECK_EQUAL(nr_inliers, kNumPoints);

  BOOST_CHECK_SMALL((poses_est[1].tvec - poses_orig[1].tvec).norm(), 1e-9);

  const Vector4d relative_rot_est = ConcatenateQuaternions(
      InvertQuaternion(poses_est[1].qvec), poses_est[0].qvec);
  const Vector4d relative_rot_orig = ConcatenateQuaternions(
      InvertQuaternion(poses_orig[1].qvec), poses_orig[0].qvec);

  BOOST_CHECK_SMALL((relative_rot_est - relative_rot_orig).norm(), 1e-9);
}
