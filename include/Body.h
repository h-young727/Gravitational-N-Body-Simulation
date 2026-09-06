#pragma once

// Represents a single gravitational body
struct Body
{
    double x, y, z;
    double vx, vy, vz;
    double mass;
};

// Net force acting on body per timestep
struct Force
{
    double fx, fy, fz;
};
