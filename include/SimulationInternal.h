#pragma once

#include <vector>

#include "Body.h"
#include "SimConfig.h"

// Exposes Simulation.cpp's internals so Main.cpp can record body trajectories
void step(std::vector<Body>& bodies, std::vector<Force>& forces, const SimConfig& config, int numThreads);
