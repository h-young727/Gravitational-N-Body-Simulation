#pragma once

#include <vector>

#include "Body.h"
#include "SimConfig.h"

std::vector<Body> simulateGPU(std::vector<Body> bodies, const SimConfig& config = SimConfig{});
