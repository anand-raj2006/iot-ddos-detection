import argparse
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score


# Lightweight but meaningful feature subset (max 10 features).
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
		print(f"Skipping {file_name}: 'label' column not found")
		return False, False

	labels = df["label"].astype(str)
	unique_labels = labels.unique().tolist()
	print(f"{file_name} unique labels: {unique_labels}")

	has_benign = labels.apply(is_benign_label).any()
	has_attack = (~labels.apply(is_benign_label)).any()
	print(f"{file_name} -> has_benign={has_benign}, has_attack={has_attack}")
	return has_benign, has_attack


def load_data(folder_path, nrows_per_file=10000):
	csv_files = get_csv_files(folder_path)
	benign_or_mixed_frames = []
	attack_only_frames = []

	benign_or_mixed_files = []
	attack_only_files = []

	print(f"Found {len(csv_files)} CSV files in: {folder_path}")
	for file_path in csv_files:
		try:
			df_part = pd.read_csv(file_path, nrows=nrows_per_file)
		except Exception as exc:
			print(f"Skipping {file_path.name} due to read error: {exc}")
			continue

		has_benign, has_attack = inspect_file_labels(df_part, file_path.name)

		if not has_benign and not has_attack:
			continue

		if has_benign:
			benign_or_mixed_frames.append(df_part)
			benign_or_mixed_files.append(file_path.name)
		else:
			attack_only_frames.append(df_part)
			attack_only_files.append(file_path.name)

	print("\nFile grouping summary:")
	print(f"Total files used: {len(benign_or_mixed_files) + len(attack_only_files)}")
	print(f"Files with benign (or mixed): {len(benign_or_mixed_files)}")
	print(f"Attack-only files: {len(attack_only_files)}")

	if benign_or_mixed_files:
		print("Benign/mixed files used:", benign_or_mixed_files)
	if attack_only_files:
		print("Attack-only files available:", attack_only_files)

	if not benign_or_mixed_frames:
		raise ValueError(
			"No files containing Benign/Normal labels were found. "
			"Please include benign/normal CSV files."
		)

	# Ensure benign presence first, then append attack-only files for broader attack coverage.
	selected_frames = benign_or_mixed_frames + attack_only_frames

	if not selected_frames:
		raise ValueError(f"No data could be loaded from: {folder_path}")

	combined = pd.concat(selected_frames, ignore_index=True)

	if "label" not in combined.columns:
		raise ValueError("Combined data does not contain 'label' column.")

	print(f"Combined dataset shape: {combined.shape}")
	print("Raw label distribution (top 20):")
	print(combined["label"].astype(str).value_counts().head(20))
	return combined


def to_binary_label(label_value):
	# 0 for benign/normal, 1 for all attacks.
	return 0 if is_benign_label(label_value) else 1


def validate_target_distribution(y_data, dataset_name):
	print(f"Final class distribution for {dataset_name}:")
	print(y_data.value_counts())

	if y_data.nunique() < 2:
		raise ValueError("Dataset contains only one class. Please check file selection.")


def preprocess_data(df, feature_columns, dataset_name):
	missing = [c for c in feature_columns + ["label"] if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns: {missing}")

	working_df = df[feature_columns + ["label"]].copy()

	# Convert labels to binary target: 0 = benign, 1 = attack.
	working_df["target"] = working_df["label"].apply(to_binary_label)

	for col in feature_columns:
		working_df[col] = pd.to_numeric(working_df[col], errors="coerce")

	working_df = working_df.dropna(subset=feature_columns + ["target"])

	x_data = working_df[feature_columns]
	y_data = working_df["target"]

	print(f"Data shape after preprocessing: X={x_data.shape}, y={y_data.shape}")
	validate_target_distribution(y_data, dataset_name)

	return x_data, y_data


def train_model(x_train, y_train):
	model = RandomForestClassifier(
		n_estimators=200,
		class_weight="balanced",
		random_state=42,
		n_jobs=-1,
	)
	model.fit(x_train, y_train)
	return model


def evaluate_model(model, x_test, y_test):
	predictions = model.predict(x_test)
	accuracy = accuracy_score(y_test, predictions)
	macro_f1 = f1_score(y_test, predictions, average="macro", zero_division=0)

	print(f"\nAccuracy: {accuracy:.4f}")
	print(f"Macro F1-score: {macro_f1:.4f}")
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
	archive_base = r"c:\Users\DELL\Documents\Project1\archive\CICIOT23"
	default_train = f"{archive_base}\\train"
	default_test = f"{archive_base}\\test"
	default_val = f"{archive_base}\\validation"

	parser = argparse.ArgumentParser(
		description="Phase 1: IoT DDoS detection model training (Random Forest)."
	)
	parser.add_argument(
		"--train_folder",
		type=str,
		default=default_train,
		help=f"Folder path containing training CSV files (default: {default_train}).",
	)
	parser.add_argument(
		"--test_folder",
		type=str,
		default=default_test,
		help=f"Folder path containing test CSV files (default: {default_test}).",
	)
	parser.add_argument(
		"--val_folder",
		type=str,
		default=default_val,
		help=f"Validation folder to merge into training data (default: {default_val}).",
	)
	parser.add_argument(
		"--nrows_per_file",
		type=int,
		default=10000,
		help="Number of rows to load from each CSV file (default: 10000).",
	)
	parser.add_argument(
		"--model_out",
		type=str,
		default="model.pkl",
		help="Output path for saved model file (default: model.pkl).",
	)
	return parser.parse_args()


def main():
	args = parse_args()

	print("\n--- Loading Training Data ---")
	train_df = load_data(args.train_folder, nrows_per_file=args.nrows_per_file)

	if args.val_folder and Path(args.val_folder).exists():
		print("\n--- Loading Validation Data (for training merge) ---")
		val_df = load_data(args.val_folder, nrows_per_file=args.nrows_per_file)
		train_df = pd.concat([train_df, val_df], ignore_index=True)
		print(f"Merged training+validation shape: {train_df.shape}")
	elif args.val_folder:
		print(f"\nNote: Validation folder not found at {args.val_folder}, skipping merge.")

	x_train, y_train = preprocess_data(train_df, FEATURES, dataset_name="training")

	print("\n--- Applying SMOTE for class balancing ---")
	print(f"Training data before SMOTE: {x_train.shape[0]} rows")
	print(f"Class distribution before SMOTE:")
	print(y_train.value_counts())

	smote = SMOTE(
		sampling_strategy={0: 25000, 1: 25000},
		random_state=42,
		k_neighbors=5,
	)
	x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

	print(f"\nTraining data after SMOTE: {x_train_smote.shape[0]} rows")
	print(f"Class distribution after SMOTE:")
	print(pd.Series(y_train_smote).value_counts())

	print("\n--- Training Random Forest Model ---")
	model = train_model(x_train_smote, y_train_smote)

	print("\n--- Loading Test Data ---")
	test_df = load_data(args.test_folder, nrows_per_file=args.nrows_per_file)
	x_test, y_test = preprocess_data(test_df, FEATURES, dataset_name="test")

	print("\n--- Evaluating Model ---")
	evaluate_model(model, x_test, y_test)

	print("\n--- Saving Model ---")
	save_model(model, args.model_out)


if __name__ == "__main__":
	main()