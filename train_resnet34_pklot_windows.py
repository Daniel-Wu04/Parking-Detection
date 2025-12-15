from __future__ import annotations

import json
import re
import time
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


# ============================================================
# CONFIG: UPDATE THESE PATHS FOR YOUR MACHINE
# ============================================================

SPLIT_UFPR04_TRAIN = Path(r"C:\Users\2022w\Desktop\ML Parking\PKLot\UFPR04_train.txt")
SPLIT_UFPR04_TEST  = Path(r"C:\Users\2022w\Desktop\ML Parking\PKLot\UFPR04_test.txt")

SPLIT_UFPR05_TRAIN = Path(r"C:\Users\2022w\Desktop\ML Parking\PKLot\UFPR05_train.txt")
SPLIT_UFPR05_TEST  = Path(r"C:\Users\2022w\Desktop\ML Parking\PKLot\UFPR05_test.txt")

SPLIT_PUC_TRAIN    = Path(r"C:\Users\2022w\Desktop\ML Parking\PKLot\PUC_train.txt")
SPLIT_PUC_TEST     = Path(r"C:\Users\2022w\Desktop\ML Parking\PKLot\PUC_test.txt")

# Root folder that contains UFPR04 / UFPR05 / PUC folders
PKLOT_SEGMENTED_ROOT = Path(r"C:\Users\2022w\Desktop\ML Parking\PKLot\PKLotSegmented")

# Where to save model checkpoints + caches
OUT_DIR = Path(r"C:\Users\2022w\Desktop\ML Parking\checkpoints")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------
# Speed knobs (3080-friendly)
# -------------------------
BATCH_SIZE = 128                 # try 128 (your VRAM looked low). If OOM -> 96 or 64.
DATALOADER_WORKERS = 4           # Windows-safe start. (Try 6 later if stable.)
PERSISTENT_WORKERS = False       # keep False on Windows unless everything is stable

EPOCHS_FROZEN = 1
EPOCHS_FINETUNE = 5

PRINT_MISSING_LIMIT = 25         # keep low to avoid spam
PROGRESS_EVERY_BATCHES = 200     # print progress inside epoch every N batches (helps confirm it's working)

# Cache the resolved sample lists so you don't re-scan the disk every run
USE_SAMPLE_CACHE = True
TRAIN_CACHE = OUT_DIR / "train_samples_cache.json"
TEST_CACHE  = OUT_DIR / "test_samples_cache.json"


# ============================================================
# GLOBAL SPEED SETTINGS (CUDA)
# ============================================================
# TF32 = free speed on Ampere (RTX 30xx) for matmul/conv; great for classification
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


# ============================================================
# DEVICE
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = (DEVICE.type == "cuda")

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True


# ============================================================
# PATH REPAIR (for known split typos)
# ============================================================
def _remove_windows_duplicate_suffix(name: str) -> str:
    # 'abc(2).jpg' -> 'abc.jpg'
    return re.sub(r"\(\d+\)(?=\.[^.]+$)", "", name)

def _repair_rel_path(rel_path: str) -> List[str]:
    """
    Generate a small deterministic set of candidates.
    Fixes observed issues:
      - Sunnny -> Sunny
      - remove '(2)' etc before extension
    """
    rel_path = rel_path.strip().replace("\\", "/")
    candidates = [rel_path]

    # folder typo
    candidates.append(rel_path.replace("Sunnny/", "Sunny/"))
    candidates.append(rel_path.replace("/Sunnny/", "/Sunny/"))

    # filename duplicate suffix
    p = Path(rel_path)
    fixed_name = _remove_windows_duplicate_suffix(p.name)
    if fixed_name != p.name:
        candidates.append(str(p.with_name(fixed_name)).replace("\\", "/"))
        candidates.append(str(p.with_name(fixed_name)).replace("\\", "/").replace("Sunnny/", "Sunny/"))
        candidates.append(str(p.with_name(fixed_name)).replace("\\", "/").replace("/Sunnny/", "/Sunny/"))

    # de-dup
    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out

def resolve_existing(segmented_root: Path, rel_path: str) -> Path | None:
    for cand_rel in _repair_rel_path(rel_path):
        abs_path = segmented_root / Path(cand_rel)
        if abs_path.exists():
            return abs_path
    return None


# ============================================================
# DATASET (with caching)
# ============================================================
class PKLotSplitDataset(Dataset):
    """
    Split lines like:
        PUC/Sunny/2012-09-17/Occupied/xxx.jpg 1
    """
    def __init__(self,
                 segmented_root: Path,
                 split_files: List[Path],
                 transform=None,
                 cache_path: Path | None = None):
        self.segmented_root = Path(segmented_root)
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        if not self.segmented_root.exists():
            raise FileNotFoundError(
                f"PKLOT_SEGMENTED_ROOT does not exist:\n  {self.segmented_root}\n"
                f"It must be the folder containing PUC/ UFPR04/ UFPR05/"
            )

        # Try cache first (huge speedup on reruns)
        if USE_SAMPLE_CACHE and cache_path is not None and cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            # validate quickly: root must match, and at least one path exists
            if data.get("root") == str(self.segmented_root.resolve()) and data.get("samples"):
                self.samples = [(p, int(y)) for p, y in data["samples"]]
                # quick sanity: if first doesn't exist, ignore cache
                if Path(self.samples[0][0]).exists():
                    print(f"✅ Loaded sample cache: {cache_path.name} ({len(self.samples)} samples)")
                    return
                else:
                    print(f"⚠️ Cache paths not valid anymore, rebuilding: {cache_path.name}")

        t0 = time.time()
        missing_count = 0
        printed_missing = 0
        repaired_count = 0

        for sf in split_files:
            sf = Path(sf)
            if not sf.exists():
                raise FileNotFoundError(f"Split file not found: {sf}")

            with sf.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 2:
                        continue

                    rel_path, label_str = parts

                    # Try original and repaired variants
                    found = resolve_existing(self.segmented_root, rel_path)
                    if found is not None:
                        # count repair usage (if original join missing)
                        orig = self.segmented_root / Path(rel_path.strip().replace("\\", "/"))
                        if not orig.exists():
                            repaired_count += 1
                        self.samples.append((str(found), int(label_str)))
                    else:
                        missing_count += 1
                        if printed_missing < PRINT_MISSING_LIMIT:
                            print(f"⚠️ Missing image (unresolved): {self.segmented_root / Path(rel_path.replace('\\', '/'))}")
                            printed_missing += 1

        dt = time.time() - t0
        print(f"Loaded {len(self.samples)} samples from {len(split_files)} split files in {dt:.1f}s.")
        if repaired_count:
            print(f"Repaired paths loaded: {repaired_count}")
        if missing_count:
            print(f"Total missing images (not loaded): {missing_count}")

        if len(self.samples) == 0:
            raise RuntimeError(
                "0 images were resolved. This almost always means PKLOT_SEGMENTED_ROOT is wrong.\n"
                f"Current root: {self.segmented_root.resolve()}"
            )

        # Save cache for next run
        if USE_SAMPLE_CACHE and cache_path is not None:
            payload = {
                "root": str(self.segmented_root.resolve()),
                "created": time.time(),
                "samples": self.samples,
            }
            with cache_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f)
            print(f"✅ Wrote sample cache: {cache_path.name}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# ============================================================
# TRANSFORMS
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ============================================================
# MODEL
# ============================================================
def get_resnet34(frozen: bool = True) -> nn.Module:
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    if frozen:
        for p in model.parameters():
            p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model.to(DEVICE)


# ============================================================
# TRAIN / EVAL
# ============================================================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        preds = model(x).argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return (correct / total) if total > 0 else 0.0


def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: optim.Optimizer,
                    scheduler=None,
                    scaler: torch.amp.GradScaler | None = None) -> float:
    """
    Scheduler is stepped AFTER optimizer.step(), which avoids the warning you saw.
    """
    model.train()
    total_loss = 0.0
    n = 0
    criterion = nn.CrossEntropyLoss()

    for batch_i, (x, y) in enumerate(loader, start=1):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if USE_AMP:
            assert scaler is not None
            with torch.amp.autocast(device_type="cuda", enabled=True):
                out = model(x)
                loss = criterion(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)      # optimizer step happens here
            scaler.update()
        else:
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        # IMPORTANT: scheduler after optimizer step
        if scheduler is not None:
            scheduler.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        n += bs

        if PROGRESS_EVERY_BATCHES and (batch_i % PROGRESS_EVERY_BATCHES == 0):
            avg = total_loss / max(n, 1)
            print(f"  batch {batch_i}/{len(loader)} | avg_loss {avg:.4f}")

    return total_loss / max(n, 1)


# ============================================================
# MAIN
# ============================================================
def run_training():
    # Print device ONCE (inside main process)
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_files = [SPLIT_PUC_TRAIN, SPLIT_UFPR04_TRAIN, SPLIT_UFPR05_TRAIN]
    test_files  = [SPLIT_PUC_TEST,  SPLIT_UFPR04_TEST,  SPLIT_UFPR05_TEST]

    # Build datasets (cached)
    train_ds = PKLotSplitDataset(PKLOT_SEGMENTED_ROOT, train_files, transform=train_transform, cache_path=TRAIN_CACHE)
    test_ds  = PKLotSplitDataset(PKLOT_SEGMENTED_ROOT, test_files,  transform=test_transform,  cache_path=TEST_CACHE)

    print("Train size:", len(train_ds))
    print("Test size :", len(test_ds))
    print("First train sample:", train_ds.samples[0][0])

    pin = (DEVICE.type == "cuda")

    # Windows-safe DataLoader config
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=DATALOADER_WORKERS,
        pin_memory=pin,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=2 if DATALOADER_WORKERS > 0 else None
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=DATALOADER_WORKERS,
        pin_memory=pin,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=2 if DATALOADER_WORKERS > 0 else None
    )

    # -------------------------
    # Stage 1: Frozen head training
    # -------------------------
    model = get_resnet34(frozen=True)
    scaler = torch.amp.GradScaler(enabled=USE_AMP)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.95, 0.999))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS_FROZEN,
        pct_start=0.3,
        anneal_strategy="cos",
        cycle_momentum=False
    )

    print("\n--- Stage 1: Frozen training ---")
    for epoch in range(EPOCHS_FROZEN):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler)
        acc = evaluate(model, test_loader)
        dt = time.time() - t0
        print(f"[Frozen] Epoch {epoch+1}/{EPOCHS_FROZEN} | Loss: {loss:.4f} | Test Acc: {acc*100:.2f}% | {dt/60:.1f} min")
        torch.save(model.state_dict(), OUT_DIR / f"pklot_resnet34_frozen_epoch{epoch+1}.pth")

    # -------------------------
    # Stage 2: Fine-tuning
    # -------------------------
    print("\n--- Stage 2: Fine-tuning ---")
    for p in model.parameters():
        p.requires_grad = True

    body_params, head_params = [], []
    for n, p in model.named_parameters():
        (head_params if "fc" in n else body_params).append(p)

    base_lr = 1e-4
    optimizer = optim.Adam(
        [
            {"params": body_params, "lr": base_lr / 10, "betas": (0.95, 0.999)},
            {"params": head_params, "lr": base_lr,      "betas": (0.95, 0.999)},
        ]
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[base_lr / 10, base_lr],
        steps_per_epoch=len(train_loader),
        epochs=EPOCHS_FINETUNE,
        pct_start=0.3,
        anneal_strategy="cos",
        cycle_momentum=False
    )

    for epoch in range(EPOCHS_FINETUNE):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler)
        acc = evaluate(model, test_loader)
        dt = time.time() - t0
        print(f"[FT] Epoch {epoch+1}/{EPOCHS_FINETUNE} | Loss: {loss:.4f} | Test Acc: {acc*100:.2f}% | {dt/60:.1f} min")
        torch.save(model.state_dict(), OUT_DIR / f"pklot_resnet34_finetune_epoch{epoch+1}.pth")

    print("\n✅ Training complete.")
    print(f"Saved checkpoints to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    mp.freeze_support()  # Windows multiprocessing safety
    run_training()
