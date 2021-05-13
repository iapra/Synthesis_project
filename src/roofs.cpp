/*
  SYNTHESIS PROJECT.2021
*/

#include <iostream>
#include <sstream>
#include <fstream>
#include <roofs.h>

void roof::roof_obstacles()  {
    // CLEAN INPUT
    // retrieve obj equation of planes

    // Discard points too close to planes (parameter = distance treshold)

    // CLUSTERING THE OBSTACLE POINTS
    // build k-d tree

    // CONVEX-HULLS
    // Project conv-hull to planes

    // Compute roof plane area

    // Compute area of projected obstacle(s) per plane (??)

    // Deduce percentage off per roof plane

    // Add/store this attribute value for each face
}

void roof::write_json(std::string filepath) {
    // TODO: not done in cpp or use a library ?
}

bool roof::read_ply (std::string filepath) {

    std::cout << "\nReading file: " << filepath << std::endl;
    std::ifstream infile(filepath.c_str(), std::ifstream::in);
    if (!infile)
    {
        std::cerr << "Input file not found.\n";
        return false;
    }
    std::string cursor;

    // start reading the header
    std::getline(infile, cursor);
    if (cursor != "ply") {
        std::cerr << "Magic ply keyword not found\n";
        return false;
    };

    std::getline(infile, cursor);
    if (cursor != "format ascii 1.0") {
        std::cerr << "Incorrect ply format\n";
        return false;
    };

    // read the remainder of the header
    std::string line = "";
    int vertex_count = 0;
    bool expectVertexProp = false, foundNonVertexElement = false;
    int property_count = 0;
    std::vector<std::string> property_names;
    int pos_x = -1, pos_y = -1, pos_z = -1, pos_segment_id = -1;

    while (line != "end_header") {
        std::getline(infile, line);
        std::istringstream linestream(line);

        linestream >> cursor;

        // read vertex element and properties
        if (cursor == "element") {
            linestream >> cursor;
            if (cursor == "vertex") {
                // check if this is the first element defined in the file. If not exit the function
                if (foundNonVertexElement) {
                    std::cerr << "vertex element is not the first element\n";
                    return false;
                };

                linestream >> vertex_count;
                expectVertexProp = true;
            } else {
                foundNonVertexElement = true;
            }
        } else if (expectVertexProp) {
            if (cursor != "property") {
                expectVertexProp = false;
            } else {
                // read property type
                linestream >> cursor;
                if (cursor.find("float") != std::string::npos || cursor == "double") {
                    // read property name
                    linestream >> cursor;
                    if (cursor == "x") {
                        pos_x = property_count;
                    } else if (cursor == "y") {
                        pos_y = property_count;
                    } else if (cursor == "z") {
                        pos_z = property_count;
                    }
                    ++property_count;
                }
                else if (cursor.find("uint") != std::string::npos || cursor == "int") {
                    // read property name
                    linestream >> cursor;
                    if (cursor == "segment_id") {
                        pos_segment_id = property_count;
                    }
                    ++property_count;
                }
            }
        }
    }

    // check if we were able to locate all the coordinate properties
    if ( pos_x == -1 || pos_y == -1 || pos_z == -1) {
        std::cerr << "Unable to locate x, y and z vertex property positions\n";
        return false;
    };
    // read the vertex properties
    for (int vi = 0; vi < vertex_count; ++vi) {
        //std::cout << vi << std::endl;
        std::getline(infile, line);
        std::istringstream linestream(line);

        double x{},y{},z{};
        int sid{};
        for (int pi = 0; pi < property_count; ++pi) {
            linestream >> cursor;
            if ( pi == pos_x ) {
                x = std::stod(cursor);
            } else if ( pi == pos_y ) {
                y = std::stod(cursor);
            } else if ( pi == pos_z ) {
                z = std::stod(cursor);
            } else if ( pi == pos_segment_id ) {
                sid = std::stoi(cursor);
            }
        }
        auto p = Point{x, y, z};
        _input_points.push_back(p);
    }

    std::cout << "Number of points read from .ply file: " << _input_points.size() << std::endl;

    return true;
}

bool roof::read_obj (std::string filepath) {
    //TODO (cf previous assignments?)
}