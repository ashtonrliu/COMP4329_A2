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

def create_index_to_label_dictionary(cfg):
    """
    A dictionary which transforms labels from [1, 19] to an index used for one-hot encoding
    """
    dictionary = {}
    index = 0

    for label in cfg["labels"]["classes"]:
        dictionary[index] = label
        index += 1
    
    return dictionary


# ── 4. glue everything together ─────────────────────────────────────────
if __name__ == "__main__":
    # load config & dataframe
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    df  = extract_df(cfg["test_filename"], cfg["data"]["basepath"])
    test_samples = create_all_samples(df, cfg, is_training=False)

    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, cfg["labels"]["num_of_labels"])   # 18- or 19-way head
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.to(device).eval()

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)
    val_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_ds = MultiLabelDataset(
        test_samples,       # ImageRecord objects with .label = None
        transform=val_tfms,
        with_labels=False
    )

    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    model.eval()
    sigmoid = torch.nn.Sigmoid()
    pred_rows = []

    IDX_TO_LABEL = create_index_to_label_dictionary(cfg)

    with torch.no_grad():
        for imgs, ids in test_loader:
            logits = model(imgs.to(device))
            probs  = sigmoid(logits).cpu()          # [B, num_classes]

            for p, img_id in zip(probs, ids):
                idxs = (p >= 0.5).nonzero(as_tuple=False).flatten().tolist()
                label_string = " ".join(IDX_TO_LABEL[i] for i in idxs)
                pred_rows.append({"ImageID": img_id + ".jpg", "Labels": label_string})

    # write out CSV (same as earlier)
    import pandas as pd
    pd.DataFrame(pred_rows).to_csv("submission.csv", index=False)

    # # model + transforms
    # model, train_tfms, val_tfms = setup_training()

    # # build ONE full dataset (no cache) then split 90/10
    # full_ds = MultiLabelDataset(samples, transform=train_tfms, cache=False)
    # val_len = int(0.10 * len(full_ds))
    # train_len = len(full_ds) - val_len
    # train_ds, val_ds = random_split(full_ds, [train_len, val_len],
    #                                 generator=torch.Generator().manual_seed(42))

    # # IMPORTANT: val set should use val transforms (no flip)
    # val_ds = type(full_ds)([full_ds.records[i] for i in val_ds.indices],
    #                        transform=val_tfms, cache=False)

    # pin_mem = device.type == "cuda"
    # train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
    #                           num_workers=4, pin_memory=pin_mem)
    # val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,
    #                           num_workers=4, pin_memory=pin_mem)

    # run_training(model, train_loader, val_loader)
