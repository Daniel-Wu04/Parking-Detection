from pathlib import Path

# -----------------------------
# UPDATED PATHS TO SPLIT FILES
# -----------------------------
TRAIN_FILES = [
    Path(r"C:\Users\2022w\Desktop\ML Parking\PKLot\UFPR04_train.txt"),
    Path(r"C:\Users\2022w\Desktop\ML Parking\PKLot\UFPR05_train.txt"),
    Path(r"C:\Users\2022w\Desktop\ML Parking\PKLot\PUC_train.txt"),
]

TEST_FILES = [
    Path(r"C:\Users\2022w\Desktop\ML Parking\PKLot\UFPR04_test.txt"),
    Path(r"C:\Users\2022w\Desktop\ML Parking\PKLot\UFPR05_test.txt"),
    Path(r"C:\Users\2022w\Desktop\ML Parking\PKLot\PUC_test.txt"),
]

OUT_TRAIN = Path("combined_train.txt")
OUT_TEST  = Path("combined_test.txt")


def combine(files, out_path):
    total_lines = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for f in files:
            if not f.exists():
                raise FileNotFoundError(f"Missing file: {f}")
            with open(f, "r", encoding="utf-8", errors="ignore") as infile:
                lines = infile.readlines()
                out.writelines(lines)
                total_lines += len(lines)
                print(f"Added {len(lines)} lines from {f.name}")
    return total_lines


def main():
    print("Combining TRAIN split files...")
    train_count = combine(TRAIN_FILES, OUT_TRAIN)

    print("\nCombining TEST split files...")
    test_count = combine(TEST_FILES, OUT_TEST)

    print("\n=== DONE ===")
    print(f"Combined TRAIN lines: {train_count}")
    print(f"Combined TEST  lines: {test_count}")
    print("\nOutput files written:")
    print(f"  {OUT_TRAIN.resolve()}")
    print(f"  {OUT_TEST.resolve()}")


if __name__ == "__main__":
    main()
