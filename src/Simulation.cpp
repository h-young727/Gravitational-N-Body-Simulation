#include "Simulation.h"
#include "SimulationInternal.h"

#include <cmath>
#include <omp.h>

static void computeForces(const std::vector<Body>& bodies, std::vector<Force>& forces, const SimConfig& config, int numThreads)
{
    int n = static_cast<int>(bodies.size());

    #pragma omp parallel for schedule(static) num_threads(numThreads)
    for (int i = 0; i < n; i++)
    {
        const Body& bi = bodies[i];
        double fx = 0.0, fy = 0.0, fz = 0.0;

        for (int j = 0; j < n; j++)
        {
            if (i == j)
            {
                continue;
            }

            const Body& bj = bodies[j];
            double dx = bj.x - bi.x;
            double dy = bj.y - bi.y;
            double dz = bj.z - bi.z;
            double distSq = dx * dx + dy * dy + dz * dz + config.forcePadding * config.forcePadding;
            double dist = std::sqrt(distSq);
            double force = (config.gravitationalConstant * bi.mass * bj.mass) / distSq;

            fx += force * (dx / dist);
            fy += force * (dy / dist);
            fz += force * (dz / dist);
        }

        forces[i] = {fx, fy, fz};
    }
}

static void reflect(double& pos, double& vel, double bound)
{
    if (pos < 0.0)
    {
        pos = 0.0;
        vel *= -1.0;
    }
    else if (pos > bound)
    {
        pos = bound;
        vel *= -1.0;
    }
}

static void updatePositions(std::vector<Body>& bodies, const std::vector<Force>& forces, const SimConfig& config)
{
    for (std::size_t i = 0; i < bodies.size(); i++)
    {
        Body& body = bodies[i];
        const Force& force = forces[i];

        body.vx += (force.fx / body.mass) * config.timestep;
        body.vy += (force.fy / body.mass) * config.timestep;
        body.vz += (force.fz / body.mass) * config.timestep;

        body.x += body.vx * config.timestep;
        body.y += body.vy * config.timestep;
        body.z += body.vz * config.timestep;

        reflect(body.x, body.vx, config.boundsWidth);
        reflect(body.y, body.vy, config.boundsHeight);
        reflect(body.z, body.vz, config.boundsDepth);
    }
}

void step(std::vector<Body>& bodies, std::vector<Force>& forces, const SimConfig& config, int numThreads)
{
    computeForces(bodies, forces, config, numThreads);
    updatePositions(bodies, forces, config);
}

std::vector<Body> simulate(std::vector<Body> bodies, const SimConfig& config, int numThreads)
{
    std::vector<Force> forces(bodies.size());

    for (int t = 0; t < config.numSteps; t++)
    {
        step(bodies, forces, config, numThreads);
    }

    return bodies;
}
