#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "InitialConditions.h"
#include "SimConfig.h"
#include "Simulation.h"

static constexpr int benchmarkSteps = 100;
static const std::vector<int> bodyCounts = {64, 512, 1024, 2048, 4096, 8192, 16384, 32768};
static const std::vector<int> threadCounts = {1, 2, 4, 8, 16};

struct Result
{
    std::string version;
    int bodies;
    int threads;
    double runtimeSeconds;
};

static double timeRun(int numBodies, int numThreads)
{
    SimConfig config;
    config.numBodies = numBodies;
    config.numSteps = benchmarkSteps;

    std::vector<Body> bodies = makeRandomBodies(config);

    const auto start = std::chrono::steady_clock::now();
    simulate(std::move(bodies), config, numThreads);
    const auto end = std::chrono::steady_clock::now();

    return std::chrono::duration<double>(end - start).count();
}

static std::vector<Result> runSweep()
{
    std::vector<Result> results;

    for (int bodies : bodyCounts)
    {
        for (int threads : threadCounts)
        {
            std::cout << "  cpu bodies=" << bodies << " threads=" << threads << "\n";
            std::string version = (threads == 1) ? "sequential" : "parallel";
            results.push_back({version, bodies, threads, timeRun(bodies, threads)});
        }
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
        file << r.version << "," << r.bodies << "," << r.threads << "," << r.runtimeSeconds << "\n";
    }

    return true;
}

static void printTable(const std::vector<Result>& results)
{
    std::cout << "\nCPU benchmark (" << benchmarkSteps << " steps)\n"
              << std::string(38, '-') << "\n"
              << std::right << std::setw(10) << "bodies" << std::setw(10) << "threads" << std::setw(18) << "runtime (s)" << "\n"
              << std::string(38, '-') << "\n";

    for (const Result& r : results)
    {
        std::cout << std::setw(10) << r.bodies << std::setw(10) << r.threads << std::setw(18) << r.runtimeSeconds << "\n";
    }
}

int main()
{
    const std::filesystem::path outputDir = "../output";
    std::filesystem::create_directories(outputDir);

    std::cout << std::fixed << std::setprecision(4);

    std::vector<Result> results = runSweep();
    printTable(results);

    const std::filesystem::path csvPath = outputDir / "bench_cpu.csv";

    if (!writeCsv(results, csvPath))
    {
        std::cerr << "Error: could not write " << csvPath.string() << "\n";
        return 1;
    }

    std::cout << "\nWrote " << csvPath.string() << "\n";

    return 0;
}
