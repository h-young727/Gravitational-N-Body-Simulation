#include <chrono>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

#include "Simulation.cuh"
#include "InitialConditions.h"
#include "SimConfig.h"

static void warmUpCuda()
{
    cudaFree(nullptr);
}

static constexpr int benchmarkSteps = 100;
static const std::vector<int> bodyCounts = {64, 512, 1024, 2048, 4096, 8192, 16384, 32768};

struct Result
{
    int bodies;
    double runtimeSeconds;
};

static double timeRun(int numBodies)
{
    SimConfig config;
    config.numBodies = numBodies;
    config.numSteps = benchmarkSteps;

    std::vector<Body> bodies = makeRandomBodies(config);

    const auto start = std::chrono::steady_clock::now();
    simulateGPU(std::move(bodies), config);
    const auto end = std::chrono::steady_clock::now();

    return std::chrono::duration<double>(end - start).count();
}

static std::vector<Result> runSweep()
{
    std::vector<Result> results;

    for (int bodies : bodyCounts)
    {
        std::cout << "  gpu bodies=" << bodies << "\n";
        results.push_back({bodies, timeRun(bodies)});
    }

    return results;
}

static bool writeCsv(const std::vector<Result>& results, const std::filesystem::path& path)
{
    std::ofstream file(path);

    if (!file.is_open())
    {
        return false;
    }

    file << "version,bodies,threads,runtime\n";

    for (const Result& r : results)
    {
        file << "gpu," << r.bodies << ",0," << r.runtimeSeconds << "\n";
    }

    return true;
}

static void printTable(const std::vector<Result>& results)
{
    std::cout << "\nGPU benchmark (" << benchmarkSteps << " steps)\n"
              << std::string(28, '-') << "\n"
              << std::right << std::setw(10) << "bodies" << std::setw(18) << "runtime (s)" << "\n"
              << std::string(28, '-') << "\n";

    for (const Result& r : results)
    {
        std::cout << std::setw(10) << r.bodies << std::setw(18) << r.runtimeSeconds << "\n";
    }
}

int main()
{
    const std::filesystem::path outputDir = "../output";
    std::filesystem::create_directories(outputDir);

    std::cout << std::fixed << std::setprecision(4);

    warmUpCuda();
    std::vector<Result> results = runSweep();
    printTable(results);

    const std::filesystem::path csvPath = outputDir / "bench_gpu.csv";

    if (!writeCsv(results, csvPath))
    {
        std::cerr << "Error: could not write " << csvPath.string() << "\n";
        return 1;
    }

    std::cout << "\nWrote " << csvPath.string() << "\n";

    return 0;
}
