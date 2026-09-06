#include "Simulation.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do \
    { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) \
        { \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            std::exit(EXIT_FAILURE); \
        } \
    } \
    while (0)

__global__ void computeForcesKernel(const Body* bodies, Force* forces, int n, SimConfig config)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
    {
        return;
    }

    Body bi = bodies[i];
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int j = 0; j < n; j++)
    {
        if (j == i)
        {
            continue;
        }

        Body bj = bodies[j];
        double dx = bj.x - bi.x;
        double dy = bj.y - bi.y;
        double dz = bj.z - bi.z;
        double distSq = dx * dx + dy * dy + dz * dz + config.forcePadding * config.forcePadding;
        double dist = sqrt(distSq);
        double force = (config.gravitationalConstant * bi.mass * bj.mass) / distSq;

        fx += force * (dx / dist);
        fy += force * (dy / dist);
        fz += force * (dz / dist);
    }

    forces[i] = {fx, fy, fz};
}

__device__ void reflect(double& pos, double& vel, double bound)
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

__global__ void updatePositionsKernel(Body* bodies, const Force* forces, int n, SimConfig config)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
    {
        return;
    }

    Body body = bodies[i];
    Force force = forces[i];

    body.vx += (force.fx / body.mass) * config.timestep;
    body.vy += (force.fy / body.mass) * config.timestep;
    body.vz += (force.fz / body.mass) * config.timestep;

    body.x += body.vx * config.timestep;
    body.y += body.vy * config.timestep;
    body.z += body.vz * config.timestep;

    reflect(body.x, body.vx, config.boundsWidth);
    reflect(body.y, body.vy, config.boundsHeight);
    reflect(body.z, body.vz, config.boundsDepth);

    bodies[i] = body;
}

static int gridSize(int n, int blockSize)
{
    return (n + blockSize - 1) / blockSize;
}

std::vector<Body> simulateGPU(std::vector<Body> bodies, const SimConfig& config)
{
    int n = static_cast<int>(bodies.size());
    int blockSize = 256;
    int numBlocks = gridSize(n, blockSize);

    Body* deviceBodies;
    Force* deviceForces;

    Body* pinnedBodies;
    CUDA_CHECK(cudaMallocHost(&pinnedBodies, n * sizeof(Body)));
    std::copy(bodies.begin(), bodies.end(), pinnedBodies);

    CUDA_CHECK(cudaMalloc(&deviceBodies, n * sizeof(Body)));
    CUDA_CHECK(cudaMalloc(&deviceForces, n * sizeof(Force)));
    CUDA_CHECK(cudaMemcpy(deviceBodies, pinnedBodies, n * sizeof(Body), cudaMemcpyHostToDevice));

    for (int t = 0; t < config.numSteps; t++)
    {
        computeForcesKernel<<<numBlocks, blockSize>>>(deviceBodies, deviceForces, n, config);
        CUDA_CHECK(cudaGetLastError());

        updatePositionsKernel<<<numBlocks, blockSize>>>(deviceBodies, deviceForces, n, config);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpy(pinnedBodies, deviceBodies, n * sizeof(Body), cudaMemcpyDeviceToHost));
    std::copy(pinnedBodies, pinnedBodies + n, bodies.begin());

    CUDA_CHECK(cudaFreeHost(pinnedBodies));
    CUDA_CHECK(cudaFree(deviceBodies));
    CUDA_CHECK(cudaFree(deviceForces));

    return bodies;
}
