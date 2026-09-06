#include "InitialConditions.h"

#include <random>

std::vector<Body> makeRandomBodies(const SimConfig& config, std::uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> x(0.0, config.boundsWidth);
    std::uniform_real_distribution<double> y(0.0, config.boundsHeight);
    std::uniform_real_distribution<double> z(0.0, config.boundsDepth);
    std::uniform_real_distribution<double> mass(config.minMass, config.maxMass);

    std::vector<Body> bodies(config.numBodies);

    for (Body& body : bodies)
    {
        body = {x(rng), y(rng), z(rng), 0.0, 0.0, 0.0, mass(rng)};
    }

    return bodies;
}
