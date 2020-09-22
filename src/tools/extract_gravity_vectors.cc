#include "base/image.h"
#include "base/reconstruction.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/option_manager.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <Eigen/Dense>

using namespace colmap;

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::string reconstruction_path;
  std::string images_path;

  OptionManager options;
  options.AddRequiredOption("reconstruction_path", &reconstruction_path);
  options.AddRequiredOption("images_path", &images_path);
  options.Parse(argc, argv);

  Reconstruction reconstruction;
  reconstruction.Read(reconstruction_path);

  bool first_file = true;

  for (const auto& ipair : reconstruction.Images()) {
    const Image& image = ipair.second;
    std::string root, ext;
    SplitFileExtension(image.Name(), &root, &ext);
    // might break if images_path is empty... ¯\_(ツ)_/¯
    const std::string gravity_file_name =
        EnsureTrailingSlash(images_path) + root + ".txt";

    if (first_file && ExistsFile(gravity_file_name)) {
      // If gravity files exist but not for the first file then what are you
      // even doing?

      std::cout
          << "Warning: this will overwrite existing .txt files in the images "
             "path. Confirm with 'y'.\n";
      char confirm_char;
      std::cin >> confirm_char;

      if (confirm_char != 'y') {
        return EXIT_FAILURE;
      }
    }
    first_file = false;

    const Eigen::Vector3d gravity =
        Eigen::Quaterniond(image.Qvec()) * Eigen::Vector3d(0, 1, 0);

    std::ofstream of(gravity_file_name);
    of << std::setprecision(9) << gravity.transpose() << '\n';
  }

  return EXIT_SUCCESS;
}