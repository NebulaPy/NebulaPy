import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SOFT_MIN, SOFT_MAX = 0.3, 2.0
HARD_MIN, HARD_MAX = 2.0, 10.0

input_file = "/Users/tony/Desktop/CWBs-NEMOv1/Post-Processing/Dominant-Lines/wr140_NEMO_d07e13_d2l6n128_DominantLines.txt"
output_dir = "/Users/tony/Desktop/CWBs-NEMOv1/Post-Processing/Dominant-Lines"
output_file = os.path.join(output_dir, "Soft&HardXrayList.txt")
os.makedirs(output_dir, exist_ok=True)

# -------------------------------------------------
# Energy conversion
# -------------------------------------------------
def angstrom_to_keV(w):
    return 12.39841984 / w


steps = {}
current_step = None
current_time = None

# -------------------------------------------------
# Read file
# -------------------------------------------------
with open(input_file) as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        if line.startswith("Step:"):
            m = re.search(r"Step:\s*(\d+)\s*\|\s*Simulation Time:\s*([0-9.eE+\-]+)", line)
            if m:
                current_step = int(m.group(1))
                current_time = float(m.group(2))
                steps[current_step] = {
                    "time": current_time,
                    "soft": [],
                    "hard": [],
                    "other": []
                }
            continue

        if line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 4 or current_step is None:
            continue

        try:
            element = parts[0]
            ion = parts[1]
            wavelength_str = parts[2]
            luminosity = float(parts[3])

            wavelength = float(wavelength_str)
            energy = angstrom_to_keV(wavelength)

            line_name = f"{element} {ion} {wavelength_str}"
            row = (line_name, wavelength, luminosity, energy)

            if SOFT_MIN <= energy < SOFT_MAX:
                steps[current_step]["soft"].append(row)
            elif HARD_MIN <= energy <= HARD_MAX:
                steps[current_step]["hard"].append(row)
            else:
                steps[current_step]["other"].append(row)

        except Exception:
            continue

# -------------------------------------------------
# Sort by luminosity
# -------------------------------------------------
for step in steps:
    for band in ["soft", "hard", "other"]:
        steps[step][band].sort(key=lambda x: x[2], reverse=True)

# -------------------------------------------------
# Write output text file
# -------------------------------------------------
with open(output_file, "w") as out:
    for step in sorted(steps):
        time = steps[step]["time"]
        header = f"Step: {step} | Simulation Time: {time:.6e} (s)"
        print(header)
        out.write(header + "\n\n")

        def print_block(title, data):
            if not data:
                return

            print(title)
            out.write(title + "\n")

            col = f"{'Line':<24}{'Luminosity':>15}{'Energy (keV)':>15}"
            print(col)
            out.write(col + "\n")

            line_sep = "-" * 55
            print(line_sep)
            out.write(line_sep + "\n")

            for line_name, wavelength, lum, energy in data:
                row = f"{line_name:<24}{lum:>15.6e}{energy:>15.4f}"
                print(row)
                out.write(row + "\n")

            print()
            out.write("\n")

        print_block("Soft X-ray lines", steps[step]["soft"])
        print_block("Hard X-ray lines", steps[step]["hard"])
        print_block("Other bright lines", steps[step]["other"])

        sep = "=" * 70
        print(sep)
        print()
        out.write(sep + "\n\n")

print("Output file:")
print(output_file)

# -------------------------------------------------
# Plot SOFT and HARD X-ray lines
# -------------------------------------------------
XMIN = None   # example: 1.5
XMAX = 20     # example: 8.0

YMIN = 1e31   # example: 1e30
YMAX = None   # example: 1e35

for step in sorted(steps):

    time = steps[step]["time"]

    soft_lines = steps[step]["soft"]
    hard_lines = steps[step]["hard"]

    # -------------------------------------------------
    # Apply wavelength filter if limits provided
    # -------------------------------------------------
    if XMIN is not None or XMAX is not None:

        def in_range(w):
            if XMIN is not None and w < XMIN:
                return False
            if XMAX is not None and w > XMAX:
                return False
            return True

        soft_lines = [row for row in soft_lines if in_range(row[1])]
        hard_lines = [row for row in hard_lines if in_range(row[1])]

    if not soft_lines and not hard_lines:
        continue

    fig, ax = plt.subplots(figsize=(16, 8))

    # -------------------------------------------------
    # Soft X-ray
    # -------------------------------------------------
    if soft_lines:

        soft_w = [row[1] for row in soft_lines]
        soft_l = [row[2] for row in soft_lines]

        ax.scatter(
            soft_w,
            soft_l,
            color="blue",
            s=45,
            label="Soft X-ray (0.3–2 keV)"
        )

        for line_name, w, lum, energy in soft_lines:
            ax.annotate(
                line_name,
                (w, lum),
                xytext=(0, 12),
                textcoords="offset points",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=8
            )

    # -------------------------------------------------
    # Hard X-ray
    # -------------------------------------------------
    if hard_lines:

        hard_w = [row[1] for row in hard_lines]
        hard_l = [row[2] for row in hard_lines]

        ax.scatter(
            hard_w,
            hard_l,
            color="red",
            s=45,
            label="Hard X-ray (2–10 keV)"
        )

        for line_name, w, lum, energy in hard_lines:
            ax.annotate(
                line_name,
                (w, lum),
                xytext=(0, 12),
                textcoords="offset points",
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=8
            )

    # -------------------------------------------------
    # Axis settings
    # -------------------------------------------------
    ax.set_yscale("log")

    ax.set_xlabel("Wavelength (Å)", fontsize=16)
    ax.set_ylabel("Luminosity", fontsize=16)

    ax.set_title(
        f"Soft and Hard X-ray Lines | Simulation Time = {time:.3e} s",
        fontsize=18
    )

    # Apply X-axis limits
    if XMIN is not None or XMAX is not None:
        ax.set_xlim(XMIN, XMAX)

    # Apply Y-axis limits
    if YMIN is not None or YMAX is not None:
        ax.set_ylim(YMIN, YMAX)

    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()

    plot_file = os.path.join(
        output_dir,
        f"SoftHard_XrayLines_step_{step:04d}.png"
    )

    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {plot_file}")