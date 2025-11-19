import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# scaling plot
results_csv = "results.csv"
df_res = pd.read_csv(results_csv)

plt.figure(figsize=(14, 7))
bodies_list = sorted(df_res["bodies"].unique())
colors = plt.cm.viridis_r(np.linspace(0, 1, len(bodies_list)))

for color, b in zip(colors, bodies_list):
    # plot cpu data
    sub_cpu = df_res[(df_res["bodies"] == b) & (df_res["threads"] > 0)]
    if not sub_cpu.empty:
        sub_cpu = sub_cpu.sort_values(by="threads")
        plt.plot(sub_cpu["threads"], sub_cpu["runtime"], marker='o', color=color, label=f"{b} bodies (CPU)")
        if 1 in sub_cpu["threads"].values:
            y = sub_cpu[sub_cpu["threads"] == 1]["runtime"].values[0]
            plt.text(1, y * 1.12, "sequential", fontsize=9, ha='center')

    # plot gpu data
    sub_gpu = df_res[(df_res["bodies"] == b) & (df_res["threads"] == 0)]
    if not sub_gpu.empty:
        y_gpu = sub_gpu["runtime"].values[0]
        plt.plot([0], [y_gpu], marker='s', markersize=8, linestyle='None', color=color, label=f"{b} bodies (GPU)")
        plt.text(0, y_gpu * 1.12, "GPU", fontsize=9, ha='center')

plt.xlabel("Number of threads (0 = GPU)")
plt.ylabel("Execution time (s)")
plt.title("N-Body Simulation Scaling")
plt.xticks([0, 1, 2, 4, 8, 16])
plt.yscale("log")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend(title="Bodies", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig("scaling.png", dpi=300)
plt.close()

# simulation animation
sim_csv = "nbody_seq_output.csv"
df_sim = pd.read_csv(sim_csv)

unique_ids = df_sim["id"].unique()[:20]
num_to_plot = len(unique_ids)

steps = sorted(df_sim["step"].unique())
num_steps = len(steps)

subsample_rate = 5
frame_indices = np.arange(0, num_steps, subsample_rate)
num_anim_frames = len(frame_indices)

trajectories = {}
for body_id in unique_ids:
    sub = df_sim[df_sim["id"] == body_id].sort_values("step")
    trajectories[body_id] = sub[["x","y"]].values

fig, ax = plt.subplots(figsize=(6,6))
all_coords = np.vstack(list(trajectories.values()))
ax.set_xlim(all_coords[:,0].min(), all_coords[:,0].max())
ax.set_ylim(all_coords[:,1].min(), all_coords[:,1].max())
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title(f"N-Body Simulation ({num_to_plot} bodies)")

scat = ax.scatter([], [], s=40, color='blue', zorder=2)

def update(frame_idx):
    frame = frame_indices[frame_idx]
    xs, ys = [], []
    for body_id in unique_ids:
        coords = trajectories[body_id]
        xs.append(coords[frame,0])
        ys.append(coords[frame,1])
    scat.set_offsets(np.column_stack([xs, ys]))
    ax.set_title(f"Step {steps[frame]} (showing {num_to_plot} bodies)")
    return [scat]

anim = FuncAnimation(fig, update, frames=num_anim_frames, interval=100, blit=True)
anim.save("nbody_simulation.gif", writer=PillowWriter(fps=30))
plt.close()
