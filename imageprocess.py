from __future__ import annotations

import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
from skimage.feature import hog


# -----------------------------
# PATHS
# -----------------------------
TRAIN_ROOT = Path(r"PKLot_Combined\train")
TEST_ROOT  = Path(r"PKLot_Combined\test")

CACHE_DIR = Path("cache_features")
CACHE_DIR.mkdir(exist_ok=True)
TRAIN_CACHE = CACHE_DIR / "train_hog_64.npz"
TEST_CACHE  = CACHE_DIR / "test_hog_64.npz"


# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = (64, 64)  # (width, height)
LABEL_MAP = {"Empty": 0, "Occupied": 1}

# Your machine: Ryzen 9 7900X + 32GB
MAX_WORKERS = 12

# Chunking prevents huge future lists and keeps RAM stable
CHUNK_SIZE = 4000

# Print progress every N processed images
PROGRESS_EVERY = 2000

# HOG parameters (consistent with what you used before)
HOG_PARAMS = dict(
    orientations=9,
    pixels_per_cell=(8, 8),
    cells_per_block=(2, 2),
    block_norm="L2-Hys"
)


# -----------------------------
# HELPERS
# -----------------------------
def gather_jobs(root: Path) -> list[tuple[str, int]]:
    """Collect (image_path, label) for all images under root based on folder name."""
    if not root.exists():
        raise FileNotFoundError(f"Missing folder: {root.resolve()}")

    jobs: list[tuple[str, int]] = []
    for dirpath, _, filenames in os.walk(root):
        label_name = Path(dirpath).name
        if label_name not in LABEL_MAP:
            continue

        label = LABEL_MAP[label_name]
        for fname in filenames:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                jobs.append((str(Path(dirpath) / fname), label))

    return jobs


def featurize_one(job: tuple[str, int]) -> tuple[np.ndarray, int] | None:
    """Worker: read image -> grayscale -> resize -> HOG -> (feature, label)."""
    path_str, label = job

    img = cv2.imread(path_str)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE)

    feat = hog(gray, **HOG_PARAMS).astype(np.float32)
    return feat, label


def build_features_parallel(root: Path) -> tuple[np.ndarray, np.ndarray]:
    jobs = gather_jobs(root)
    n = len(jobs)
    print(f"\nFound {n} images under: {root.resolve()}")
    if n == 0:
        raise RuntimeError(f"No images found under {root.resolve()}")

    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    done = 0
    bad = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        # Submit in chunks to avoid huge memory spikes
        for start in range(0, n, CHUNK_SIZE):
            chunk = jobs[start:start + CHUNK_SIZE]
            futures = [ex.submit(featurize_one, j) for j in chunk]

            for fut in as_completed(futures):
                out = fut.result()
                if out is None:
                    bad += 1
                    continue

                feat, label = out
                X_list.append(feat)
                y_list.append(label)

                done += 1
                if done % PROGRESS_EVERY == 0:
                    dt = time.time() - t0
                    rate = done / dt if dt > 0 else 0.0
                    print(f"  Processed {done}/{n} | rate ~ {rate:.1f} imgs/sec | bad reads: {bad}")

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    print(f"Finished {root.name}: X={X.shape}, y={y.shape}, bad_reads={bad}")
    print(f"  Empty: {int(np.sum(y==0))}  Occupied: {int(np.sum(y==1))}")
    return X, y


def save_cache(cache_path: Path, X: np.ndarray, y: np.ndarray):
    print(f"💾 Saving cache: {cache_path.resolve()}")
    np.savez_compressed(cache_path, X=X, y=y)


def main():
    print(f"Config: IMG_SIZE={IMG_SIZE}, MAX_WORKERS={MAX_WORKERS}, CHUNK_SIZE={CHUNK_SIZE}")
    print("This script WILL read images and compute HOG features.\n")

    # -------- TRAIN --------
    if TRAIN_CACHE.exists():
        print(f"✅ Train cache already exists: {TRAIN_CACHE.resolve()}")
        print("   (Delete it if you want to rebuild.)")
    else:
        print("Building TRAIN cache...")
        X_train, y_train = build_features_parallel(TRAIN_ROOT)
        save_cache(TRAIN_CACHE, X_train, y_train)

    # -------- TEST --------
    if TEST_CACHE.exists():
        print(f"\n✅ Test cache already exists: {TEST_CACHE.resolve()}")
        print("   (Delete it if you want to rebuild.)")
    else:
        print("\nBuilding TEST cache...")
        X_test, y_test = build_features_parallel(TEST_ROOT)
        save_cache(TEST_CACHE, X_test, y_test)

    print("\n✅ DONE: feature caches are ready.")
    print(f"  Train cache: {TRAIN_CACHE}")
    print(f"  Test cache : {TEST_CACHE}")


if __name__ == "__main__":
    main()
