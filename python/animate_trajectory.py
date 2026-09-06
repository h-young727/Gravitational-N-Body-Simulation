from collections import deque
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from PIL import Image

TRAJECTORY_CSV = Path("output/trajectory.csv")
BODIES_CSV = Path("output/bodies.csv")
OUTPUT_GIF = Path("output/nbody_simulation.gif")
SUBSAMPLE_RATE = 5
FPS = 30
MIN_MARKER_SIZE = 20.0
MAX_MARKER_SIZE = 600.0
TRAIL_LENGTH = 200


def load_trajectories(trajectory_csv: Path) -> tuple[dict[int, np.ndarray], list[int]]:
    df = pd.read_csv(trajectory_csv)
    steps = sorted(df["step"].unique())
    trajectories = {body_id: group.sort_values("step")[["x", "y"]].values for body_id, group in df.groupby("id")}
    return trajectories, steps


def load_masses(bodies_csv: Path) -> dict[int, float]:
    df = pd.read_csv(bodies_csv)
    return dict(zip(df["id"], df["mass"]))


def marker_sizes_from_masses(masses: dict[int, float]) -> dict[int, float]:
    values = np.array(list(masses.values()))
    mass_range = values.max() - values.min()

    if mass_range == 0:
        midpoint = (MIN_MARKER_SIZE + MAX_MARKER_SIZE) / 2
        return dict.fromkeys(masses.keys(), midpoint)

    normalized = (values - values.min()) / mass_range
    sizes = MIN_MARKER_SIZE + normalized * (MAX_MARKER_SIZE - MIN_MARKER_SIZE)
    return dict(zip(masses.keys(), sizes))


def body_colors(num_bodies: int) -> list[tuple[float, float, float, float]]:
    palette = plt.cm.tab10.colors
    return [(*palette[i % len(palette)], 1.0) for i in range(num_bodies)]


def update_trail(collection: LineCollection, history: deque, color: tuple[float, float, float, float]) -> None:
    if len(history) < 2:
        return

    points = np.array(history).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    num_segments = len(segments)
    segment_colors = np.tile(color, (num_segments, 1))
    segment_colors[:, 3] = np.linspace(0.05, 0.8, num_segments)

    collection.set_segments(segments)
    collection.set_color(segment_colors)


def add_mass_legend(ax: plt.Axes, body_ids: list[int], colors: list[tuple[float, float, float, float]], masses: dict[int, float]) -> None:
    order = sorted(range(len(body_ids)), key=lambda i: masses[body_ids[i]], reverse=True)
    handles = [Line2D([], [], marker="o", linestyle="", color=colors[i], markeredgecolor="black") for i in order]
    labels = [f"{masses[body_ids[i]]:.2e} kg" for i in order]
    ax.legend(handles, labels, title="Bodies, by mass", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)


def render_frames(trajectories: dict[int, np.ndarray], steps: list[int], masses: dict[int, float]) -> Iterator[Image.Image]:
    body_ids = list(trajectories.keys())
    frame_indices = np.arange(0, len(steps), SUBSAMPLE_RATE)
    marker_sizes = marker_sizes_from_masses(masses)
    sizes = [marker_sizes[body_id] for body_id in body_ids]
    colors = body_colors(len(body_ids))

    fig, ax = plt.subplots(figsize=(9, 6))
    all_coords = np.vstack([trajectories[body_id] for body_id in body_ids])
    ax.set_xlim(all_coords[:, 0].min(), all_coords[:, 0].max())
    ax.set_ylim(all_coords[:, 1].min(), all_coords[:, 1].max())
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.suptitle(f"{len(body_ids)}-Body Simulation")
    add_mass_legend(ax, body_ids, colors, masses)
    fig.tight_layout(rect=[0, 0, 0.78, 1])

    trails = [LineCollection([], linewidths=2, zorder=2) for _ in body_ids]
    for trail in trails:
        ax.add_collection(trail)
    histories = [deque(maxlen=TRAIL_LENGTH) for _ in body_ids]

    scat = ax.scatter([], [], edgecolors="black", linewidths=0.6, alpha=0.85, zorder=3)
    scat.set_sizes(sizes)
    scat.set_color(colors)

    for frame in frame_indices:
        positions = np.array([trajectories[body_id][frame] for body_id in body_ids])
        scat.set_offsets(positions)
        ax.set_title(f"Timestep: {steps[frame]}")

        for history, position, trail, color in zip(histories, positions, trails, colors):
            history.append(position)
            update_trail(trail, history, color)

        fig.canvas.draw()
        image = Image.frombytes("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
        yield image.convert("RGB")

    plt.close(fig)


def main() -> None:
    trajectories, steps = load_trajectories(TRAJECTORY_CSV)
    masses = load_masses(BODIES_CSV)
    frames = render_frames(trajectories, steps, masses)

    # A single shared palette (from the first frame) keeps colors consistent across frames and keeps the GIF small
    first_frame = next(frames).quantize(colors=128, method=Image.Quantize.FASTOCTREE)
    quantized_frames = (frame.quantize(palette=first_frame) for frame in frames)

    first_frame.save(OUTPUT_GIF, save_all=True, append_images=quantized_frames, duration=1000 // FPS, loop=0, optimize=True)

    print(f"Wrote {OUTPUT_GIF}")


if __name__ == "__main__":
    main()
