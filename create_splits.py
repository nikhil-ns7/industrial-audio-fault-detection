import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = r"C:\Fault_Recognition\master_dataset.csv"

TRAIN_OUTPUT = r"C:\Fault_Recognition\data\splits\train.csv"
VAL_OUTPUT = r"C:\Fault_Recognition\data\splits\validation.csv"
TEST_OUTPUT = r"C:\Fault_Recognition\data\splits\test.csv"


# Load master dataset
df = pd.read_csv(INPUT)

print("Master dataset loaded")
print(f"Total recordings: {len(df)}")


# Create a combined stratification column
# This preserves both machine type and class distribution.
df["stratify_group"] = (
    df["machine"] + "_" + df["label"]
)


# 70% Train, 30% temporary
train, temp = train_test_split(
    df,
    test_size=0.30,
    stratify=df["stratify_group"],
    random_state=42
)


# Split temporary 50/50
# → 15% Validation
# → 15% Test
validation, test = train_test_split(
    temp,
    test_size=0.50,
    stratify=temp["stratify_group"],
    random_state=42
)


# Remove helper column
train = train.drop(columns=["stratify_group"])
validation = validation.drop(columns=["stratify_group"])
test = test.drop(columns=["stratify_group"])


# Save splits
train.to_csv(TRAIN_OUTPUT, index=False)
validation.to_csv(VAL_OUTPUT, index=False)
test.to_csv(TEST_OUTPUT, index=False)


# -----------------------------
# Verification
# -----------------------------

print("\n" + "=" * 60)
print("DATASET SPLIT COMPLETED")
print("=" * 60)

print(f"\nTrain      : {len(train)}")
print(f"Validation : {len(validation)}")
print(f"Test       : {len(test)}")
print(f"Total      : {len(train) + len(validation) + len(test)}")


print("\nClass distribution:")
print("\nTRAIN")
print(train["label"].value_counts())

print("\nVALIDATION")
print(validation["label"].value_counts())

print("\nTEST")
print(test["label"].value_counts())


print("\nMachine distribution:")
print("\nTRAIN")
print(train["machine"].value_counts())

print("\nVALIDATION")
print(validation["machine"].value_counts())

print("\nTEST")
print(test["machine"].value_counts())


print("\nFiles saved:")
print(TRAIN_OUTPUT)
print(VAL_OUTPUT)
print(TEST_OUTPUT)

print("=" * 60)