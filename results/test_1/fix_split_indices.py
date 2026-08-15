"""
Fix split indices for Test 1.

The original split_indices.npz was created using np.isin(y, y_train) which
matches by class label value, not row index. This produces identical arrays
for all three splits.

The correct approach: store the actual row indices from train_test_split.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

ROOT = Path(r"C:\Users\ASUS\Downloads\power-grid-fault-intelligence")

# ===========================================================================
# Load data exactly as Test 1 does
# ===========================================================================
df = pd.read_csv(ROOT / "classData.csv")
print("Dataset shape:", df.shape)

FAULT_CODE_TO_LABEL = {
    "0000": "No Fault",
    "1001": "Line-to-Line AB",
    "0110": "LL-G BC",
    "0111": "Three-Phase",
    "1011": "LG A+B-G",
    "1111": "3P-G",
}

fault_codes = df[["G", "C", "B", "A"]].astype(str).agg("".join, axis=1)
df["fault_label"] = fault_codes.map(FAULT_CODE_TO_LABEL)

RAW_FEATURES = ["Ia", "Ib", "Ic", "Va", "Vb", "Vc"]
X_raw = df[RAW_FEATURES].values
y_raw = df["fault_label"].values

le = LabelEncoder()
y = le.fit_transform(y_raw)

RANDOM_STATE = 42

# ===========================================================================
# Perform the exact same splits as Test 1
# ===========================================================================
# Step 1: Split off 15% test set
X_dev, X_test, y_dev, y_test, dev_idx, test_idx = train_test_split(
    X_raw, y, np.arange(len(X_raw)),
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=y
)

# Step 2: Split dev into train (70%) and val (15%)
X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
    X_dev, y_dev, dev_idx,
    test_size=0.15/0.85,
    random_state=RANDOM_STATE,
    stratify=y_dev
)

print("\nSplit sizes:")
print(f"  Total:   {len(X_raw)}")
print(f"  Train:   {len(X_train)} ({100*len(X_train)/len(X_raw):.1f}%)")
print(f"  Val:     {len(X_val)} ({100*len(X_val)/len(X_raw):.1f}%)")
print(f"  Test:    {len(X_test)} ({100*len(X_test)/len(X_raw):.1f}%)")

print("\nIndex shapes:")
print(f"  train_idx: {train_idx.shape}")
print(f"  val_idx:   {val_idx.shape}")
print(f"  test_idx:  {test_idx.shape}")

# Verify no overlap
train_set = set(train_idx.tolist())
val_set = set(val_idx.tolist())
test_set = set(test_idx.tolist())
assert len(train_set & val_set) == 0, "Train/Val overlap!"
assert len(train_set & test_set) == 0, "Train/Test overlap!"
assert len(val_set & test_set) == 0, "Val/Test overlap!"
print("\nNo overlap between splits: PASS")

# Verify total
assert len(train_set) + len(val_set) + len(test_set) == len(X_raw), "Missing or extra indices!"
print(f"All {len(X_raw)} indices accounted for: PASS")

# Verify data consistency
assert np.array_equal(X_raw[train_idx], X_train), "Train data mismatch!"
assert np.array_equal(X_raw[val_idx], X_val), "Val data mismatch!"
assert np.array_equal(X_raw[test_idx], X_test), "Test data mismatch!"
print("Data consistency: PASS")

# Verify class distribution
print("\nClass distribution:")
for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    dist = {le.inverse_transform([i])[0]: int(c) for i, c in enumerate(np.bincount(y_split, minlength=len(le.classes_)))}
    print(f"  {split_name}: {dist}")

# ===========================================================================
# Save corrected split indices
# ===========================================================================
split_path = ROOT / "results" / "test_1" / "split_indices.npz"
np.savez_compressed(
    split_path,
    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx,
)
print(f"\nCorrected split indices saved to: {split_path}")

# Verify the saved file
saved = np.load(split_path)
assert np.array_equal(saved["train_idx"], train_idx), "Saved train_idx mismatch!"
assert np.array_equal(saved["val_idx"], val_idx), "Saved val_idx mismatch!"
assert np.array_equal(saved["test_idx"], test_idx), "Saved test_idx mismatch!"
print("Verification: saved indices match computed indices: PASS")

print("\nFirst 10 train indices:", train_idx[:10])
print("First 10 val indices:", val_idx[:10])
print("First 10 test indices:", test_idx[:10])
