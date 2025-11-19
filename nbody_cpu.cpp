#include <cstdlib>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include<omp.h>
#include <vector>

#define G 6.67430e-8
#define DT 0.1
#define NUM_STEPS 10000
#define WIDTH 1000.0
#define HEIGHT 1000.0
#define DEPTH 1000.0

struct Body { double x, y, z, vx, vy, vz, mass; };

struct Force { double fx, fy, fz; };

std::vector<Body> bodies;
std::vector<Force> forces;

void compute_forces(int num_bodies) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_bodies; ++i) {
        
        const Body& bi = bodies[i];
        double fx{0.0}, fy{0.0}, fz{0.0};

        for (int j = 0; j < num_bodies; ++j) {
            if (i == j) {
                continue;
            }

            const Body& bj = bodies[j];

            double dx = bj.x - bi.x;
            double dy = bj.y - bi.y;
            double dz = bj.z - bi.z;
            double dist = std::sqrt(dx*dx + dy*dy + dz*dz + 1e-12);
            double force = (G * bi.mass * bj.mass) / (dist * dist);
            
            fx += force * (dx / dist);
            fy += force * (dy / dist);
            fz += force * (dz / dist);
        }

        forces[i] = {fx, fy, fz};
    }
}

void update_positions(int num_bodies) {
    for (int i = 0; i < num_bodies; ++i) {

        const Force& f = forces[i];
        Body& b = bodies[i];

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
    }
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

void save_to_csv(std::ofstream &file, int step, int num_bodies) {
    for (int i = 0; i < num_bodies; ++i) {
        const Body& b = bodies[i];
        file << step << "," << i << "," << b.x << "," << b.y << "," << b.z << "\n";
    }
}

void run_simulation(int num_bodies, int num_threads) {
    std::ofstream nbody_seq_output;

    if (num_threads == 1) {
        nbody_seq_output.open("nbody_seq_output.csv");
        nbody_seq_output << "step,id,x,y,z\n";
    }

    for (int step = 0; step < NUM_STEPS; ++step) {
        compute_forces(num_bodies);
        update_positions(num_bodies);

        if (num_threads == 1) {
            save_to_csv(nbody_seq_output, step, num_bodies);
        }
    }

    if (num_threads == 1) {
        nbody_seq_output.close();
    }
}

int main(int argc, char *argv[]) {
    int num_bodies = atoi(argv[1]);
    bodies.resize(num_bodies);
    forces.resize(num_bodies);

    int num_threads = atoi(argv[2]);
    omp_set_num_threads(num_threads);

    srand(0);
    initialize_bodies(num_bodies);

    auto start = std::chrono::high_resolution_clock::now();
    run_simulation(num_bodies, num_threads);
    auto end = std::chrono::high_resolution_clock::now();
    double runtime = std::chrono::duration<double>(end - start).count();

    std::cout << "RUNTIME=" << runtime << std::endl;

    return 0;
}