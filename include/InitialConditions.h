#pragma once

#include <cstdint>
#include <vector>

#include "Body.h"
#include "SimConfig.h"

// Seed for benchmark reproducibility
std::vector<Body> makeRandomBodies(const SimConfig& config, std::uint64_t seed = 0);
