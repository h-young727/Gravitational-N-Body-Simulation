#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "InitialConditions.h"
#include "SimConfig.h"
#include "SimulationInternal.h"

int main()
{
    const std::filesystem::path outputDir = "../output";
    std::filesystem::create_directories(outputDir);

    const SimConfig config;

    std::vector<Body> bodies = makeRandomBodies(config);
    std::vector<Force> forces(bodies.size());

    std::ofstream bodiesFile(outputDir / "bodies.csv");

    if (!bodiesFile.is_open())
    {
        std::cerr << "Error: could not write bodies file\n";
        return 1;
    }

    bodiesFile << "id,mass\n";

    for (std::size_t i = 0; i < bodies.size(); i++)
    {
        bodiesFile << i << "," << bodies[i].mass << "\n";
    }

    std::ofstream file(outputDir / "trajectory.csv");

    if (!file.is_open())
    {
        std::cerr << "Error: could not write trajectory file\n";
        return 1;
    }

    file << "step,id,x,y,z\n";

    for (int t = 0; t < config.numSteps; t++)
    {
        // Sequential simulation for trajectory animation
        step(bodies, forces, config, 1);

        for (std::size_t i = 0; i < bodies.size(); i++)
        {
            file << t << "," << i << "," << bodies[i].x << "," << bodies[i].y << "," << bodies[i].z << "\n";
        }
    }

    std::cout << "Wrote " << (outputDir / "trajectory.csv").string() << "\n";

    return 0;
}
