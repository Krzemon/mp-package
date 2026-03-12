from pathlib import Path
import subprocess
import argparse

# ------------------------------------------------------------
# ARGUMENTY
# ------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    required=True,
    choices=["ratio", "variance", "proportion", "trials"],
    help="Tryb animacji (musi istnieć katalog img/frames_<mode>)",
)
parser.add_argument(
    "--fps",
    type=int,
    default=50,
    help="FPS animacji (domyślnie 50 = 2x szybciej niż 40 ms)",
)
args = parser.parse_args()

# ------------------------------------------------------------
# KATALOGI
# ------------------------------------------------------------
base_dir = Path(__file__).parent
image_dir = base_dir / "img" / f"frames_{args.mode}"
output_webm = base_dir / f"{args.mode}.webm"

# ------------------------------------------------------------
# SPRAWDZENIE KLATEK
# ------------------------------------------------------------
image_files = sorted(image_dir.glob("frame_*.png"))
if not image_files:
    raise RuntimeError(f"Brak obrazków w katalogu: {image_dir}")

# ------------------------------------------------------------
# FFmpeg – WebM (VP9)
# ------------------------------------------------------------
pattern = str(image_dir / "frame_*.png")

cmd = [
    "ffmpeg",
    "-y",
    # "-framerate", str(args.fps),
    "-framerate", str(1),
    "-pattern_type", "glob",
    "-i", pattern,
    "-c:v", "libvpx-vp9",
    "-pix_fmt", "yuv420p",
    "-crf", "30",
    "-b:v", "0",
    str(output_webm),
]

print("▶ Uruchamiam:")
print(" ".join(cmd))

subprocess.run(cmd, check=True)

print(f"\n✔ WebM zapisany: {output_webm}")
print(f"✔ Liczba klatek: {len(image_files)}")