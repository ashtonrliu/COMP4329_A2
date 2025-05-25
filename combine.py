import pandas as pd
from pathlib import Path

# ---------- 1.  read out.csv  ----------
out_lists = []
with Path("captions_output.csv").open() as f:
    for raw in f:
        line = raw.strip()
        if not line:                      # blank line  →  empty list
            out_lists.append([])
        else:                             # e.g. "1,19"
            nums = [int(x) for x in line.split(",") if x]
            out_lists.append(nums)

# ---------- 2.  read submission.csv  ----------
sub = pd.read_csv("image_output.csv")       # columns: ImageID, Labels

# ---------- 3.  turn the Labels column into lists ----------
def to_int_list(cell) -> list[int]:
    """
    - Works whether labels are space- or comma-separated.
    - Handles NaNs (pd.NA / float('nan')) gracefully.
    """
    if pd.isna(cell):
        return []
    # allow both commas and spaces as separators
    parts = str(cell).replace(",", " ").split()
    return [int(p) for p in parts if p]

sub["label_list"] = sub["Labels"].apply(to_int_list)

# ---------- 4.  union with the corresponding out.csv row ----------
if len(out_lists) != len(sub):
    raise ValueError(
        f"Row count mismatch: out.csv has {len(out_lists)} rows, "
        f"submission.csv has {len(sub)} rows."
    )

sub["union"] = [
    sorted(set(o) | set(s))               # keep ascending order
    for o, s in zip(out_lists, sub["label_list"])
]

# overwrite the Labels column with the merged list, space-separated
sub["Labels"] = sub["union"].apply(lambda L: " ".join(map(str, L)))

# ---------- 5.  export ----------
final_cols = sub[["ImageID", "Labels"]]
final_cols.to_csv("Predicted_labels.txt", index=False)

print("Wrote final_submission.csv")
