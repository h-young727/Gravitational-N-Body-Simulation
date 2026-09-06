#pragma once

// SI units throughout (kg, m, s)
struct SimConfig
{
    int numBodies = 10;
    double gravitationalConstant = 6.674e-11;
    double timestep = 1.0e10;
    int numSteps = 10000;
    double boundsWidth = 3.0e9;
    double boundsHeight = 3.0e9;
    double boundsDepth = 3.0e9;
    double minMass = 1.0e12;
    double maxMass = 2.0e12;

    // Prevents the force acting on a body from going to infinity
    double forcePadding = 1.0e8;
};
