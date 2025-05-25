# ─── train.py (only the parts below the imports) ─────────────────────────
import yaml, random
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet50
from torchvision import transforms
from sklearn.metrics import f1_score

from dataset import extract_df, MultiLabelDataset
from data_transform import create_all_samples

# ── 1. choose device once ────────────────────────────────────────────────
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print("Using device:", device)


# ── 2. helper: returns model + train-time transforms ─────────────────────
def setup_training():
    torch.hub.set_dir("checkpoints")

    # build ResNet-50 backbone, load ImageNet weights
    state_path = "checkpoints/resnet50_v2.pth"
    state      = torch.load(state_path, map_location="cpu")

    model = resnet50(weights=None)
    model.load_state_dict(state)          # 1 000-class params
    model.fc = nn.Linear(model.fc.in_features, 18)   # 18 way head

    # data transforms
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)
    train_tfms = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tfms   = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return model, train_tfms, val_tfms


# ── 3. main training routine (unchanged except val_loader arg) ───────────
def run_training(model, train_loader, val_loader):
    EPOCH_WARM, EPOCH_FINE = 5, 20
    LR_HEAD, LR_BACKBONE, WEIGHT_DECAY = 1e-3, 1e-4, 1e-4
    BEST_PATH = Path("best_model.pth")

    criterion = nn.BCEWithLogitsLoss()
    head_params     = list(model.fc.parameters())
    backbone_params = [p for n, p in model.named_parameters() if "fc" not in n]

    # freeze backbone for warm-up
    for p in backbone_params:
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [{"params": head_params,     "lr": LR_HEAD},
         {"params": backbone_params, "lr": 0.0}],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCH_FINE, eta_min=1e-6
    )

    model.to(device)
    best_micro = 0.0
    total_epochs = EPOCH_WARM + EPOCH_FINE

    for epoch in range(total_epochs):
        # ----- training phase -----
        model.train()
        running = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            loss = criterion(model(imgs), lbls)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running += loss.item() * imgs.size(0)

        # unfreeze backbone after warm-up
        if epoch + 1 == EPOCH_WARM:
            print("🔓  unfreezing backbone …")
            for p in backbone_params: p.requires_grad = True
            for g in optimizer.param_groups: g["lr"] = LR_BACKBONE

        # ----- validation phase -----
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                probs = torch.sigmoid(model(imgs.to(device))).cpu()
                preds.append((probs >= 0.5).float()); trues.append(lbls)
        preds, trues = torch.cat(preds), torch.cat(trues)
        micro_f1 = f1_score(trues, preds, average="micro", zero_division=0)
        macro_f1 = f1_score(trues, preds, average="macro", zero_division=0)
        print(f"Epoch {epoch+1:02}/{total_epochs} | "
              f"loss {running/len(train_loader.dataset):.4f} | "
              f"micro-F1 {micro_f1:.3f} | macro-F1 {macro_f1:.3f}")

        # checkpoint
        if micro_f1 > best_micro:
            best_micro = micro_f1
            torch.save(model.state_dict(), BEST_PATH)
            print(f"✅  saved new best to {BEST_PATH}")

        if epoch + 1 > EPOCH_WARM: scheduler.step()

    print(f"🏁  done — best micro-F1 = {best_micro:.3f}")


# ── 4. glue everything together ─────────────────────────────────────────
if __name__ == "__main__":
    # load config & dataframe
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    df  = extract_df(cfg["train_filename"], cfg["data"]["basepath"])
    samples = create_all_samples(df, cfg)

    # model + transforms
    model, train_tfms, val_tfms = setup_training()

    # build ONE full dataset (no cache) then split 90/10
    full_ds = MultiLabelDataset(samples, transform=train_tfms, cache=False)
    val_len = int(0.10 * len(full_ds))
    train_len = len(full_ds) - val_len
    train_ds, val_ds = random_split(full_ds, [train_len, val_len],
                                    generator=torch.Generator().manual_seed(42))

    # IMPORTANT: val set should use val transforms (no flip)
    val_ds = type(full_ds)([full_ds.records[i] for i in val_ds.indices],
                           transform=val_tfms, cache=False)

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=4, pin_memory=pin_mem)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,
                              num_workers=4, pin_memory=pin_mem)

    run_training(model, train_loader, val_loader)
