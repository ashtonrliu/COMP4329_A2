

from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import csv


def append_none_to_caption(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append non-null entries from the 'None' column into the caption column, then drop the 'None' column.
    """
    caption_col = "Caption"
    extra_col = None
    sep = " "

    if extra_col is None:
        if None in df.columns:
            extra_col = None
        elif 'None' in df.columns:
            extra_col = 'None'
        else:
            raise KeyError("Could not find a column named None or 'None' in your DataFrame.")

    # make sure captions are strings
    df[caption_col] = df[caption_col].astype(str)

    # build a Series of the extra text, safely capturing sep in the lambda’s default
    extras = df[extra_col].apply(
        lambda val, sep=sep: (
            '' if pd.isna(val)
            else sep.join(str(item).strip() for item in (val if isinstance(val, (list, tuple)) else [val]))
        )
    )

    # only append where there actually is some extra text
    mask = extras.ne('')
    df.loc[mask, caption_col] = df.loc[mask, caption_col] + sep + extras[mask]

    return df.drop(columns=[extra_col])

def extract_df(filename, basepath):
    path = os.path.join(basepath, filename)
    with open(path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=",", quotechar='"', escapechar="\\")
        df = pd.DataFrame(reader)

    return append_none_to_caption(df)

from torch.utils.data import Dataset
import torch

class MultiLabelDataset(Dataset):
    """
    A thin wrapper around a list[ImageRecord] so it plays nicely with
    torch.utils.data.DataLoader.

    If `with_labels=True`  each __getitem__ returns
        img   : FloatTensor [3, H, W]
        label : FloatTensor [num_classes]   (multi-hot)

    If `with_labels=False` each __getitem__ returns
        img   : FloatTensor [3, H, W]
        tag   : any additional info you want – here we return the record's `id`
    """
    def __init__(
        self,
        records,
        transform=None,
        cache=False,
        with_labels=True,
    ):
        self.records     = records
        self.transform   = transform
        self.cache       = cache
        self.with_labels = with_labels

        # -------- optional RAM caching (only if labels are present) --------
        if cache:
            self._imgs = []
            self._labels_or_ids = []     # could be label vectors *or* ids
            for rec in self.records:
                img = rec.image
                if self.transform:
                    img = self.transform(img)

                self._imgs.append(img)

                if self.with_labels:
                    lab = torch.tensor(rec.one_hot_encode, dtype=torch.float32)
                    self._labels_or_ids.append(lab)
                else:                     # inference mode
                    self._labels_or_ids.append(rec.id)

    # ---------------------------------------------------------------------
    def __len__(self):
        return len(self.records)

    # ---------------------------------------------------------------------
    def __getitem__(self, idx):
        # ---- use cached tensors if requested --------------------------------
        if self.cache:
            if self.with_labels:
                return self._imgs[idx], self._labels_or_ids[idx]
            else:
                return self._imgs[idx], self._labels_or_ids[idx]   # id string

        # ---- on-the-fly processing -----------------------------------------
        rec = self.records[idx]
        img = rec.image
        if self.transform:
            img = self.transform(img)

        if self.with_labels:
            label = torch.tensor(rec.one_hot_encode, dtype=torch.float32)
            return img, label
        else:
            return img, rec.id          # keep the id so you can match outputs
