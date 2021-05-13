/*
  SYNTHESIS PROJECT.2021
*/

#include <string>
#include <vector>
#include <../external/linalg/linalg.h>

using double3 = linalg::aliases::double3;

class roof {

    struct Point : double3 {
        using double3::double3;
    };

public:
    // function to detect roof obstacles
    void roof_obstacles();

    // function to read input .PLY
    bool read_ply(std::string filepath);

    // function to write output .JSON file
    void write_json(std::string filepath);

private:
    // This variable holds the entire input point cloud after calling read_ply()
    std::vector<Point> _input_points;
};

