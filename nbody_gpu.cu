#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>

#define G 6.67430e-8
#define DT 0.1
#define NUM_STEPS 10000
#define WIDTH 1000.0
#define HEIGHT 1000.0
#define DEPTH 1000.0

#define CUDA_CHECK(call) \
do { \
cudaError_t err = call; \
if (err != cudaSuccess) { \
fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
exit(EXIT_FAILURE); \
} \
} while (0)

struct Body { double x, y, z, vx, vy, vz, mass; };

struct Force { double fx, fy, fz; };

std::vector<Body> bodies;

__global__ void compute_forces_kernel(Body* d_bodies, Force* d_forces, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N) {
        return;
    }

    Body bi = d_bodies[i];
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int j = 0; j < N; ++j) {
        if (i == j) {
            continue;
        }

        Body bj = d_bodies[j];
        double dx = bj.x - bi.x;
        double dy = bj.y - bi.y;
        double dz = bj.z - bi.z;
        double dist = sqrt(dx*dx + dy*dy + dz*dz + 1e-12);
        double F = (G * bi.mass * bj.mass) / (dist * dist);
        fx += F * dx / dist;
        fy += F * dy / dist;
        fz += F * dz / dist;
    }

    d_forces[i] = {fx, fy, fz};
}

__global__ void update_positions_kernel(Body* d_bodies, Force* d_forces, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N) {
        return;
    }

    Body b = d_bodies[i];
    Force f = d_forces[i];

    b.vx += (f.fx / b.mass) * DT;
    b.vy += (f.fy / b.mass) * DT;
    b.vz += (f.fz / b.mass) * DT;

    b.x += b.vx * DT;
    b.y += b.vy * DT;
    b.z += b.vz * DT;

    if (b.x < 0) { b.x = 0; b.vx *= -1; }
    else if (b.x > WIDTH) { b.x = WIDTH; b.vx *= -1; }

    if (b.y < 0) { b.y = 0; b.vy *= -1; }
    else if (b.y > HEIGHT) { b.y = HEIGHT; b.vy *= -1; }

    if (b.z < 0) { b.z = 0; b.vz *= -1; }
    else if (b.z > DEPTH) { b.z = DEPTH; b.vz *= -1; }

    d_bodies[i] = b;
}

void initialize_bodies(int num_bodies) {
    for (int i = 0; i < num_bodies; ++i) {
        bodies[i] = {static_cast<double>(rand() % 1000),
                     static_cast<double>(rand() % 1000),
                     static_cast<double>(rand() % 1000),
                     0.0, 0.0, 0.0,
                     1e9 + static_cast<double>(rand() % static_cast<int>(1e9))};
    }
}

void compute_forces(int num_bodies, Body* d_bodies, Force* d_forces, int blockSize, int gridSize) {
    compute_forces_kernel<<<gridSize, blockSize>>>(d_bodies, d_forces, num_bodies);
    CUDA_CHECK(cudaGetLastError());
}

void update_positions(int num_bodies, Body* d_bodies, Force* d_forces, int blockSize, int gridSize) {
    update_positions_kernel<<<gridSize, blockSize>>>(d_bodies, d_forces, num_bodies);
    CUDA_CHECK(cudaGetLastError());
}

void run_simulation(int num_bodies) {
    Body* d_bodies;
    Force* d_forces;
    CUDA_CHECK(cudaMalloc(&d_bodies, num_bodies * sizeof(Body)));
    CUDA_CHECK(cudaMalloc(&d_forces, num_bodies * sizeof(Force)));

    CUDA_CHECK(cudaMemcpy(d_bodies, bodies.data(), num_bodies * sizeof(Body), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (num_bodies + blockSize - 1) / blockSize;

    for (int step = 0; step < NUM_STEPS; ++step) {
        compute_forces(num_bodies, d_bodies, d_forces, blockSize, gridSize);
        update_positions(num_bodies, d_bodies, d_forces, blockSize, gridSize);
    }

    CUDA_CHECK(cudaMemcpy(bodies.data(), d_bodies, num_bodies * sizeof(Body), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_bodies));
    CUDA_CHECK(cudaFree(d_forces));
}

int main(int argc, char* argv[]) {
    int num_bodies = atoi(argv[1]);
    bodies.resize(num_bodies);

    srand(0);
    initialize_bodies(num_bodies);

    auto start = std::chrono::high_resolution_clock::now();
    run_simulation(num_bodies);
    auto end = std::chrono::high_resolution_clock::now();
    double runtime = std::chrono::duration<double>(end - start).count();

    std::cout << "RUNTIME=" << runtime << std::endl;

    return 0;
}