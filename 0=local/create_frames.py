import sys
import os
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# ------------------------------------------------------------
# sys.path
# ------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mp_package.marchenko_pastur import (
    theoretical_eigenvalue_distribution,
    generate_eigenvalues,
)

# ------------------------------------------------------------
# ARGUMENTY
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["ratio", "variance", "proportion", "trials"],
    required=True,
)
args = parser.parse_args()

# ------------------------------------------------------------
# STYL
# ------------------------------------------------------------
rc("text.latex", preamble=r"\usepackage{lmodern} \usepackage{physics}")
plt.rcParams.update({
    "font.size": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.labelsize": 22,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.5,
    "figure.figsize": (16, 10),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "text.usetex": True,
    "font.family": "Computer Modern Roman",
})

# ------------------------------------------------------------
# PARAMETRY BAZOWE
# ------------------------------------------------------------
N_base = [40, 10]
N_total = sum(N_base)
sigma_base = [2.0, 9.0]
T_base = 300
num_trials_base = 8000
bins = 120
dist_name = "normal"

num_frames = 10

# ------------------------------------------------------------
# ZAKRESY PARAMETRÓW
# ------------------------------------------------------------
if args.mode == "ratio":
    param_values = np.linspace(200, N_total, num_frames, dtype=int)

elif args.mode == "variance":
    # ❗ sigma = 0 powoduje osobliwości → zaczynamy od >0
    sigma1 = np.linspace(0.3, 4.0, num_frames)
    sigma2 = np.linspace(12.0, 4.0, num_frames)
    param_values = list(zip(sigma1, sigma2))

elif args.mode == "proportion":
    N1 = np.linspace(10, int(0.6*N_total), num_frames, dtype=int)
    N2 = N_total - N1
    param_values = list(zip(N1, N2))

elif args.mode == "trials":
    param_values = np.linspace(500, 100000, num_frames, dtype=int)

# ------------------------------------------------------------
# KATALOG NA KLATKI
# ------------------------------------------------------------
base_dir = Path(__file__).parent
frame_dir = base_dir / "img" / f"frames_{args.mode}"
frame_dir.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# GENEROWANIE KLATEK (ODPORNE NA BŁĘDY)
# ------------------------------------------------------------
frame_idx = 0  # ⬅️ prawdziwy licznik zapisanych klatek

for i, param in enumerate(param_values):

    N_list = N_base.copy()
    sigma_squared_list = sigma_base.copy()
    T = T_base
    num_trials = num_trials_base

    if args.mode == "ratio":
        T = int(param)

    elif args.mode == "variance":
        sigma_squared_list = list(param)

    elif args.mode == "proportion":
        N_list = list(param)

    elif args.mode == "trials":
        num_trials = int(param)

    # --------------------------------------------------------
    # TEORIA + SYMULACJA (ZABEZPIECZONE)
    # --------------------------------------------------------
    try:
        x_theo, rho_theo, _ = theoretical_eigenvalue_distribution(
            N_list,
            T,
            sigma_squared_list,
            num_points=1200,
        )

        eigenvalues, _ = generate_eigenvalues(
            N_list,
            T,
            sigma_squared_list,
            num_trials,
            dist_name,
        )

    except Exception as e:
        print(f"[{args.mode}] pomijam param {i}: {e}")
        continue

    # --------------------------------------------------------
    # HISTOGRAM
    # --------------------------------------------------------
    counts, edges = np.histogram(
        eigenvalues,
        bins=bins,
        density=True,
    )
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots()

    ax.bar(
        centers,
        counts,
        width=edges[1] - edges[0],
        color="#94b694",
        edgecolor="#5fa761",
        alpha=0.7,
    )
    ax.plot(x_theo, rho_theo, color="#a72c00")

    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("PDF")
    ax.grid(True, linestyle="--", alpha=0.4)

    frame_path = frame_dir / f"frame_{frame_idx:05d}.png"
    plt.savefig(frame_path)
    plt.close()

    if frame_idx % 50 == 0:
        print(f"[{args.mode}] zapisano frame {frame_idx}")

    frame_idx += 1

# ------------------------------------------------------------
# PODSUMOWANIE
# ------------------------------------------------------------
print(f"\n✔ Zapisano {frame_idx} poprawnych klatek PNG w:")
print(f"  {frame_dir}")