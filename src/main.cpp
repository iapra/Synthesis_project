/*
  SYNTHESIS PROJECT.2021
*/

#include <iostream>
#include <filesystem>
namespace fs = std::__fs::filesystem;

//-- our trees class
#include "roofs.h"

int main () {
    roof maker;

    //-- read point cloud from input .ply file, exit if it fails
    std::string input_file = "../data/one_building.ply";
    std::string input_file2 = "../data/3d_one_building.obj";
    std::string output_file = "../data/obstacles_out.json";
    if (!maker.read_ply(input_file)) {
        return 1;
    }

    // compute and write output file
    // maker.detect_obtacles();
    // maker.read_ply(input_file);
    // maker.read_obj(input_file2);
    // maker.write_json(output_file);

    // process finished - return 0 to say all went fine
    return 0;
}