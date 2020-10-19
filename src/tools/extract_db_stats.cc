#include "base/database.h"
#include "util/option_manager.h"

#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>

using namespace colmap;

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::string db_path;
  bool stats_for_verified;

  OptionManager options;
  options.AddRequiredOption("db_path", &db_path);
  options.AddRequiredOption("verified", &stats_for_verified);
  options.Parse(argc, argv);

  Database db(db_path);

  if (!stats_for_verified){
    std::vector<std::pair<image_pair_t, FeatureMatches>> matches = db.ReadAllMatches();
    for (auto const& m : matches) {
      std::cout << m.second.size() << '\n';
    }
  } else {
    std::vector<std::pair<image_t, image_t>> image_pairs;
    std::vector<int> nr_verified_inliers;
    db.ReadTwoViewGeometryNumInliers(&image_pairs, &nr_verified_inliers);

    for (auto const& nr : nr_verified_inliers) {
      std::cout << nr << '\n';
    }
  }
}
