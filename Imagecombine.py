from pathlib import Path
import shutil

# -----------------------------
# INPUT SPLIT FILES
# -----------------------------
TRAIN_TXT = Path("combined_train.txt")
TEST_TXT  = Path("combined_test.txt")

# -----------------------------
# ORIGINAL PKLot ROOT
# -----------------------------
ORIGINAL_ROOT = Path(
    r"C:\Users\2022w\Desktop\ML Parking\PKLot\PKLotSegmented"
)

# -----------------------------
# OUTPUT ROOT
# -----------------------------
OUT_ROOT = Path("PKLot_Combined")
TRAIN_ROOT = OUT_ROOT / "train"
TEST_ROOT  = OUT_ROOT / "test"

TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
TEST_ROOT.mkdir(parents=True, exist_ok=True)


def parse_line(line: str):
    """
    Parse lines like:
      UFPR04/Sunny/2012-12-17/Occupied/xxx.jpg 1
    Returns:
      relative_path (Path), label (int)
    """
    parts = line.strip().split()
    if len(parts) != 2:
        return None, None

    rel_path, label = parts
    try:
        label = int(label)
    except ValueError:
        return None, None

    return Path(rel_path), label


def copy_preserve_structure(txt_path: Path, out_base: Path):
    copied = 0
    skipped = 0

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rel_path, label = parse_line(line)
            if rel_path is None:
                print(f"⚠️ Invalid line format, skipping: {line.strip()}")
                skipped += 1
                continue

            src = ORIGINAL_ROOT / rel_path
            if not src.exists():
                print(f"⚠️ Missing file: {src}")
                skipped += 1
                continue

            dst = out_base / rel_path
            dst.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(src, dst)
            copied += 1

    return copied, skipped


def main():
    print("Materializing PKLot combined dataset (structure preserved)")
    print("Original root:", ORIGINAL_ROOT)
    print("Output root:  ", OUT_ROOT.resolve())

    print("\nCopying TRAIN images...")
    train_copied, train_skipped = copy_preserve_structure(TRAIN_TXT, TRAIN_ROOT)

    print("\nCopying TEST images...")
    test_copied, test_skipped = copy_preserve_structure(TEST_TXT, TEST_ROOT)

    print("\n=== DONE ===")
    print(f"Train images copied: {train_copied}")
    print(f"Train images skipped: {train_skipped}")
    print(f"Test images copied:  {test_copied}")
    print(f"Test images skipped: {test_skipped}")


if __name__ == "__main__":
    main()
