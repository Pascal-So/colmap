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

#include "base/pose.h"
#include "estimators/gravity_relative_pose.h"
#include "util/random.h"

using namespace colmap;

BOOST_AUTO_TEST_CASE(Estimate) {
  SetPRNGSeed(0);

  const std::size_t kNumPoints = 3;

  std::vector<Eigen::Vector3d> points3D;
  for (std::size_t i = 0; i < kNumPoints; ++i) {
    points3D.emplace_back(RandomReal<double>(-1, 1),
                          RandomReal<double>(-1, 1),
                          RandomReal<double>(0.5, 2.5));
  }

  for (double rot_axis_z = 0; rot_axis_z < 0.5; rot_axis_z += 0.1) {
    const Eigen::Vector3d rotation_axis =
        Eigen::Vector3d(0, 1, rot_axis_z).normalized();
    GravityRelativePoseEstimator est;
    est.SetRotationAxis(rotation_axis);

    // Here we make sure to have a case where the rotation is exactly zero.
    for (double angle = -0.75; angle < 1; angle += 0.25) {
      const Eigen::Quaterniond quat(Eigen::AngleAxisd(angle, rotation_axis));
      const Eigen::Vector4d qvec = EigenQuaternionToQuaternion(quat);
      const Eigen::Matrix3d rmat = quat.toRotationMatrix();

      for (double trans_y = 0; trans_y < 0.5; trans_y += 0.1) {
        Eigen::Vector3d translation(1, trans_y, 0.1);
        translation.normalize();

        std::vector<Eigen::Vector2d> points1, points2;
        points1.reserve(kNumPoints);
        points2.reserve(kNumPoints);

        for (std::size_t i = 0; i < kNumPoints; ++i) {
          points1.push_back(points3D[i].hnormalized());
          points2.push_back((rmat * points3D[i] + translation).hnormalized());
        }

        const auto candidates = est.Estimate(points1, points2);

        BOOST_CHECK_LE(candidates.size(), 4);

        bool ok;
        int index = -1;
        for (auto c : candidates) {
          ok = true;
          ++index;

          if (std::abs(c.qvec.squaredNorm() - 1) > 1e-5)
            ok = false;

          if (c.qvec(0) < 0)
            c.qvec *= -1;

          if ((c.qvec - qvec).squaredNorm() > 1e-8)
            ok = false;

          // Translation is only computed up to a (possibly negative) factor.
          c.tvec.normalize();
          if (c.tvec.dot(translation) < 0)
            c.tvec *= -1;

          if ((c.tvec - translation).squaredNorm() > 1e-8)
            ok = false;

          if (ok)
            break;
        }

        BOOST_CHECK(ok);

        std::vector<double> residuals;
        est.Residuals(points1, points2, candidates[index], &residuals);

        for (double r : residuals) {
          BOOST_CHECK_SMALL(r, 1e-8);
        }
      }
    }
  }
}
