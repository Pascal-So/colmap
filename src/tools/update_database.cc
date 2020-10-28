#include "base/image.h"
#include "base/database.h"
#include "util/logging.h"
#include "util/misc.h"
#include "util/option_manager.h"
#include "base/image_reader.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <Eigen/Dense>

using namespace colmap;

int main(int argc, char** argv) {
  InitializeGlog(argv);

  std::string database_path;
  std::string images_path;

  OptionManager options;
  options.AddRequiredOption("database_path", &database_path);
  options.AddRequiredOption("images_path", &images_path);
  options.Parse(argc, argv);

  Database db(database_path);

  auto images = db.ReadAllImages();

  for (auto& image : images) {
    image.SetGravityPrior(ReadImageGravityPrior(EnsureTrailingSlash(images_path) + image.Name()));
    db.UpdateImage(image);
  }

  return EXIT_SUCCESS;
}
