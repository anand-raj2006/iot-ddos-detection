import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# Lightweight feature subset (10 features max)
FEATURES = [
    "flow_duration",
    "Header_Length",
    "Protocol Type",
    "Rate",
    "Srate",
    "ack_count",
    "syn_count",
    "rst_count",
    "Tot size",
    "IAT",
]

BENIGN_KEYWORDS = ("benign", "normal")


def get_csv_files(folder_path):
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Invalid folder path: {folder_path}")

    files = sorted(folder.glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in folder: {folder_path}")
    return files


def is_benign_label(label_value):
    text = str(label_value).strip().lower()
    return any(keyword in text for keyword in BENIGN_KEYWORDS)


def inspect_file_labels(df, file_name):
    if "label" not in df.columns:
        print(f"Skipping {file_name}: no 'label' column")
        return False, False

    labels = df["label"].astype(str).dropna()
    unique_labels = labels.unique().tolist()

    # Keep output readable if many labels are present.
    preview_labels = unique_labels[:15]
    if len(unique_labels) > 15:
        preview_labels.append("...")

    print(f"{file_name} unique labels: {preview_labels}")

    has_benign = labels.apply(is_benign_label).any()
    has_attack = (~labels.apply(is_benign_label)).any()

    return has_benign, has_attack


def validate_binary_target(y_data, dataset_name):
    counts = y_data.value_counts().sort_index()
    print(f"\nBinary class distribution for {dataset_name}:")
    print(counts)

    if counts.shape[0] < 2:
        raise ValueError("Dataset contains only one class. Please check file selection.")


def load_data(folder_path, nrows_per_file=10000):
    csv_files = get_csv_files(folder_path)

    selected_frames = []
    benign_files = []
    attack_only_files = []
    mixed_files = []

    print(f"\nScanning {len(csv_files)} CSV files in: {folder_path}")

    for file_path in csv_files:
        try:
            df_part = pd.read_csv(file_path, nrows=nrows_per_file)
        except Exception as exc:
            print(f"Skipping {file_path.name}: read error -> {exc}")
            continue

        has_benign, has_attack = inspect_file_labels(df_part, file_path.name)

        if not has_benign and not has_attack:
            continue

        # Keep only files that contribute to binary classification.
        selected_frames.append(df_part)

        if has_benign and has_attack:
            mixed_files.append(file_path.name)
        elif has_benign:
            benign_files.append(file_path.name)
        else:
            attack_only_files.append(file_path.name)

    if not selected_frames:
        raise ValueError(f"No usable CSV data loaded from: {folder_path}")

    if not benign_files and not mixed_files:
        raise ValueError(
            "No benign/normal samples found in loaded files. "
            "Please include files containing Benign or Normal traffic."
        )

    if not attack_only_files and not mixed_files:
        raise ValueError(
            "No attack samples found in loaded files. "
            "Please include attack-labeled CSV files."
        )

    combined = pd.concat(selected_frames, ignore_index=True)

    print("\n--- File Selection Summary ---")
    print(f"Files used: {len(selected_frames)}")
    print(f"Files with benign only: {len(benign_files)}")
    print(f"Files with attack only: {len(attack_only_files)}")
    print(f"Files with mixed labels: {len(mixed_files)}")

    if benign_files:
        print("Benign-only files:", benign_files)
    if attack_only_files:
        print("Attack-only files:", attack_only_files)
    if mixed_files:
        print("Mixed-label files:", mixed_files)

    print(f"Final combined shape: {combined.shape}")

    if "label" in combined.columns:
        print("\nRaw label distribution (top 20):")
        print(combined["label"].astype(str).value_counts().head(20))

    return combined


def to_binary_label(label_value):
    return 0 if is_benign_label(label_value) else 1


def preprocess_data(df, feature_columns, dataset_name):
    required = feature_columns + ["label"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {dataset_name}: {missing}")

    work_df = df[required].copy()

    # Convert labels to 0 (benign/normal) and 1 (attack).
    work_df["target"] = work_df["label"].apply(to_binary_label)

    for col in feature_columns:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    work_df = work_df.dropna(subset=feature_columns + ["target"])

    x_data = work_df[feature_columns]
    y_data = work_df["target"]

    print(f"\nPreprocessed {dataset_name} shape: X={x_data.shape}, y={y_data.shape}")
    print(f"Target value_counts() for {dataset_name}:")
    print(y_data.value_counts())

    validate_binary_target(y_data, dataset_name)

    return x_data, y_data


def train_model(x_train, y_train):
    class_counts = y_train.value_counts()
    minority_ratio = class_counts.min() / class_counts.max()
    print(f"\nTraining class imbalance ratio (minority/majority): {minority_ratio:.4f}")

    # class_weight='balanced' helps with skewed benign/attack counts.
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Benign (0)", "Attack (1)"],
            digits=4,
            zero_division=0,
        )
    )


def save_model(model, output_path):
    joblib.dump(model, output_path)
    print(f"\nModel saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate RandomForest on CICIoT2023 (binary benign vs attack)."
    )
    parser.add_argument("--train_folder", type=str, required=True, help="Training CSV folder path")
    parser.add_argument("--test_folder", type=str, required=True, help="Test CSV folder path")
    parser.add_argument(
        "--val_folder",
        type=str,
        default=None,
        help="Optional validation CSV folder to merge into training data",
    )
    parser.add_argument(
        "--nrows_per_file",
        type=int,
        default=10000,
        help="Rows to read from each CSV file (default: 10000)",
    )
    parser.add_argument("--model_out", type=str, default="model.pkl", help="Output model path")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n--- Loading Training Data ---")
    train_df = load_data(args.train_folder, nrows_per_file=args.nrows_per_file)

    if args.val_folder:
        print("\n--- Loading Validation Data (merged into training) ---")
        val_df = load_data(args.val_folder, nrows_per_file=args.nrows_per_file)
        train_df = pd.concat([train_df, val_df], ignore_index=True)
        print(f"Merged train+validation shape: {train_df.shape}")

    x_train, y_train = preprocess_data(train_df, FEATURES, dataset_name="training")

    print("\n--- Training Model ---")
    model = train_model(x_train, y_train)

    print("\n--- Loading Test Data ---")
    test_df = load_data(args.test_folder, nrows_per_file=args.nrows_per_file)
    x_test, y_test = preprocess_data(test_df, FEATURES, dataset_name="test")

    print("\n--- Evaluating Model ---")
    evaluate_model(model, x_test, y_test)

    print("\n--- Saving Model ---")
    save_model(model, args.model_out)


if __name__ == "__main__":
    main()
