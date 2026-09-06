#pragma once

#include <vector>

#include "Body.h"
#include "SimConfig.h"

std::vector<Body> simulate(std::vector<Body> bodies, const SimConfig& config = SimConfig{}, int numThreads = 1);
