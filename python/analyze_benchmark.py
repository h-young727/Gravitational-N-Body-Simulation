from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

CPU_CSV = Path("output/bench_cpu.csv")
GPU_CSV = Path("output/bench_gpu.csv")
OPENMP_CHART_PNG = Path("output/benchmark_openmp_scaling.png")
GPU_CHART_PNG = Path("output/benchmark_gpu_comparison.png")
SUMMARY_TXT = Path("output/benchmark_summary.txt")

CHART_SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SEQUENTIAL_RAMP_ENDPOINTS = ["#86b6ef", "#0d366b"]
OPENMP_COLOR = "#2a78d6"
GPU_COLOR = "#eb6834"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    cpu = pd.read_csv(CPU_CSV)
    gpu = pd.read_csv(GPU_CSV) if GPU_CSV.exists() else None
    return cpu, gpu


def sequential_runtime(cpu: pd.DataFrame, bodies: int) -> float:
    return cpu.loc[(cpu["version"] == "sequential") & (cpu["bodies"] == bodies), "runtime"].iloc[0]


def best_openmp(cpu: pd.DataFrame, bodies: int) -> tuple[int, float]:
    subset = cpu[(cpu["version"] == "parallel") & (cpu["bodies"] == bodies)]
    row = subset.loc[subset["runtime"].idxmin()]
    return int(row["threads"]), float(row["runtime"])


def gpu_runtime(gpu: pd.DataFrame, bodies: int) -> float:
    return gpu.loc[gpu["bodies"] == bodies, "runtime"].iloc[0]


def percent_speedup(speedup: float) -> float:
    return (speedup - 1.0) * 100.0


def sequential_ramp(num_colors: int) -> list[str]:
    cmap = mcolors.LinearSegmentedColormap.from_list("body_count_ramp", SEQUENTIAL_RAMP_ENDPOINTS)
    return [mcolors.to_hex(cmap(t)) for t in np.linspace(0.0, 1.0, num_colors)]


def style_axes(ax: plt.Axes) -> None:
    ax.set_axisbelow(True)
    ax.set_facecolor(CHART_SURFACE)
    for spine in ax.spines.values():
        spine.set_color(BASELINE)
    ax.tick_params(colors=MUTED_INK)
    ax.xaxis.label.set_color(SECONDARY_INK)
    ax.yaxis.label.set_color(SECONDARY_INK)
    ax.title.set_color(PRIMARY_INK)


def best_openmp_thread_label(cpu: pd.DataFrame, body_counts: list[int]) -> str:
    best_threads = pd.Series([best_openmp(cpu, bodies)[0] for bodies in body_counts])
    return f"Best OpenMP ({best_threads.mode()[0]} threads)"


def plot_openmp_scaling(cpu: pd.DataFrame, body_counts: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    fig.patch.set_facecolor(CHART_SURFACE)

    max_threads = cpu.loc[cpu["version"] == "parallel", "threads"].max()
    ideal = [percent_speedup(t) for t in (1, max_threads)]
    ax.plot([1, max_threads], ideal, linestyle="--", color=BASELINE, linewidth=1.5, label="Linear (ideal)")

    colors = sequential_ramp(len(body_counts))
    for bodies, color in zip(body_counts, colors):
        subset = cpu[(cpu["bodies"] == bodies) & cpu["version"].isin(["sequential", "parallel"])].sort_values("threads")
        speedup = sequential_runtime(cpu, bodies) / subset["runtime"]
        ax.plot(subset["threads"], percent_speedup(speedup), marker="o", markersize=5, linewidth=2, color=color, label=f"{bodies} bodies")

    ax.set_xlabel("OpenMP threads")
    ax.set_ylabel("% speedup over sequential")
    ax.set_title("OpenMP Scaling", loc="left")
    ax.set_xticks(sorted({1, *cpu.loc[cpu["version"] == "parallel", "threads"].unique()}))
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK)
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(OPENMP_CHART_PNG, dpi=200, facecolor=CHART_SURFACE)
    plt.close(fig)


def plot_gpu_comparison(cpu: pd.DataFrame, gpu: pd.DataFrame, body_counts: list[int]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor(CHART_SURFACE)

    best_omp_speedup, gpu_speedup = [], []
    for bodies in body_counts:
        seq = sequential_runtime(cpu, bodies)
        _, best_time = best_openmp(cpu, bodies)
        best_omp_speedup.append(percent_speedup(seq / best_time))
        gpu_speedup.append(percent_speedup(seq / gpu_runtime(gpu, bodies)))

    x = np.arange(len(body_counts))
    width = 0.35

    ax.bar(x - width / 2, best_omp_speedup, width, color=OPENMP_COLOR, label=best_openmp_thread_label(cpu, body_counts), edgecolor=PRIMARY_INK, linewidth=0.4)
    ax.bar(x + width / 2, gpu_speedup, width, color=GPU_COLOR, label="CUDA (GPU)", edgecolor=PRIMARY_INK, linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in body_counts])
    ax.set_xlabel("Bodies")
    ax.set_ylabel("% speedup over sequential")
    ax.set_title("GPU vs Best OpenMP Thread Count", loc="left")
    ax.grid(True, axis="y", color=GRIDLINE, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK)
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(GPU_CHART_PNG, dpi=200, facecolor=CHART_SURFACE)
    plt.close(fig)


def write_summary(cpu: pd.DataFrame, gpu: pd.DataFrame | None, cpu_body_counts: list[int], comparison_body_counts: list[int], gpu_only_body_counts: list[int]) -> None:
    lines = ["N-Body Benchmark Summary"]

    for bodies in cpu_body_counts:
        seq = sequential_runtime(cpu, bodies)
        lines.append(f"bodies={bodies}")
        lines.append(f"  sequential runtime: {seq:.6f} s")

        subset = cpu[(cpu["bodies"] == bodies) & (cpu["version"] == "parallel")].sort_values("threads")
        for _, row in subset.iterrows():
            threads = int(row["threads"])
            runtime = row["runtime"]
            speedup = seq / runtime
            lines.append(f"  openmp threads={threads:<3} runtime={runtime:.6f} s  speedup={speedup:.4f}x  efficiency={speedup / threads * 100:.2f}%")

        if gpu is not None and bodies in comparison_body_counts:
            gpu_time = gpu_runtime(gpu, bodies)
            best_threads, best_time = best_openmp(cpu, bodies)
            gpu_speedup = seq / gpu_time
            gpu_vs_best_omp = best_time / gpu_time
            lines.append(f"  gpu runtime={gpu_time:.6f} s  speedup vs sequential={gpu_speedup:.4f}x  speedup vs best openmp ({best_threads}t)={gpu_vs_best_omp:.4f}x")

        lines.append("")

    if gpu is None:
        lines.append(f"No GPU benchmark data found ({GPU_CSV} missing) - build with a CUDA compiler and run nbody_bench_gpu to include GPU results.")
        lines.append("")
    elif gpu_only_body_counts:
        lines.append("GPU-only sizes (no CPU sequential baseline collected):")
        for bodies in gpu_only_body_counts:
            lines.append(f"  bodies={bodies}  gpu runtime={gpu_runtime(gpu, bodies):.6f} s")
        lines.append("")

    if gpu is not None and comparison_body_counts:
        max_bodies = max(comparison_body_counts)
        seq_max = sequential_runtime(cpu, max_bodies)
        best_threads_max, best_time_max = best_openmp(cpu, max_bodies)
        gpu_time_max = gpu_runtime(gpu, max_bodies)
        lines.append(f"Headlined Result (N={max_bodies}):")
        lines.append(f"  OpenMP speedup ({best_threads_max} threads): {seq_max / best_time_max:.4f}x")
        lines.append(f"  GPU speedup: {seq_max / gpu_time_max:.4f}x")
        lines.append(f"  GPU vs best OpenMP: {best_time_max / gpu_time_max:.4f}x")
    else:
        max_bodies = max(cpu_body_counts)
        seq_max = sequential_runtime(cpu, max_bodies)
        best_threads_max, best_time_max = best_openmp(cpu, max_bodies)
        lines.append(f"Headlined Result (N={max_bodies}):")
        lines.append(f"  OpenMP speedup ({best_threads_max} threads): {seq_max / best_time_max:.4f}x")
    lines.append("")

    SUMMARY_TXT.write_text("\n".join(lines) + "\n")


def main() -> None:
    cpu, gpu = load_data()
    cpu_body_counts = sorted(cpu["bodies"].unique())

    plot_openmp_scaling(cpu, cpu_body_counts)
    print(f"Wrote {OPENMP_CHART_PNG}")

    comparison_body_counts, gpu_only_body_counts = [], []
    if gpu is not None:
        gpu_body_counts = sorted(gpu["bodies"].unique())
        comparison_body_counts = sorted(set(cpu_body_counts) & set(gpu_body_counts))
        gpu_only_body_counts = sorted(set(gpu_body_counts) - set(cpu_body_counts))
        plot_gpu_comparison(cpu, gpu, comparison_body_counts)
        print(f"Wrote {GPU_CHART_PNG}")
    else:
        print(f"No {GPU_CSV} found - skipping GPU comparison chart")

    write_summary(cpu, gpu, cpu_body_counts, comparison_body_counts, gpu_only_body_counts)
    print(f"Wrote {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
