"""
TextTrace — Final ML Experiment Pipeline
PRG4206 Machine Learning Coursework #2
INTI International University | Session Aug 2026

Purpose
-------
This is the cleaned, final training/evaluation script for the ML stage of
TextTrace. It is designed to produce the evidence needed for the later
Streamlit system and the coursework paper.

It intentionally separates:
1. AI-generated-text detection
2. Similarity analysis (implemented later in Streamlit)

The script does NOT call a score "AI plagiarism". AI likelihood and source
similarity are separate results.

Current primary models:
- TF-IDF + Logistic Regression
- TF-IDF + Calibrated Linear SVM
- XGBoost + linguistic/style features
- DistilRoBERTa
- Validation-tuned ensemble

Optional:
- DeBERTa is left as an explicit future/extension switch because it was part
  of the literature plan but is not part of the current validated result set.

Expected local project root:
    E:\\TexTrace1

The script prefers:
    E:\\TexTrace1\\data\\processed\\raid_500k.csv

If that file is not found, it can fall back to a streaming RAID sampler.
"""

# ============================================================
# 0. CONFIGURATION
# ============================================================

import os
import re
import json
import random
import time
import warnings
from pathlib import Path

SEED = 42
random.seed(SEED)

PROJECT_ROOT = Path(
    os.environ.get("TEXTTRACE_ROOT", r"E:\TexTrace1")
)

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
CHARTS_DIR = REPORTS_DIR / "charts"

for folder in [PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, CHARTS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

INPUT_CSV = PROCESSED_DIR / "raid_500k.csv"

# Fallback only if the local 500K file is missing.
FALLBACK_SAMPLE_SIZE = 500_000

# DistilRoBERTa was trained on a controlled 50K subset in the current
# validated experiment. Keep this fixed for reproducibility.
TRANSFORMER_MODEL = "distilroberta-base"
TRANSFORMER_TRAIN_LIMIT = 50_000
MAX_LENGTH = 512
TRANSFORMER_EPOCHS = 2
TRANSFORMER_BATCH_SIZE = 4
TRANSFORMER_GRAD_ACCUMULATION = 4
TRANSFORMER_LR = 2e-5

# Set True only when you intentionally want to run the optional extension.
ENABLE_DEBERTA = False

EXPECTED_DOMAINS = [
    "abstracts", "books", "news", "poetry",
    "recipes", "reddit", "reviews", "wiki"
]

print("=" * 78)
print("TEXTTRACE — FINAL ML EXPERIMENT PIPELINE")
print("=" * 78)
print("[OK] Project root:", PROJECT_ROOT)
print("[OK] Reports:", REPORTS_DIR)
print("[OK] Charts:", CHARTS_DIR)
print("[OK] Seed:", SEED)


# ============================================================
# 1. IMPORTS + PACKAGE CHECK
# ============================================================

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

np.random.seed(SEED)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("[OK] XGBoost:", xgb.__version__)
except Exception as exc:
    XGB_AVAILABLE = False
    print("[WARNING] XGBoost unavailable:", exc)

try:
    import torch
    TORCH_AVAILABLE = True
    print("[OK] PyTorch:", torch.__version__)
    print("[OK] CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[OK] GPU:", torch.cuda.get_device_name(0))
        print("[OK] CUDA:", torch.version.cuda)
except Exception as exc:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch unavailable:", exc)

warnings.filterwarnings("ignore")


# ============================================================
# 2. DATA LOADING
# ============================================================

def clean_text(text):
    """Conservative cleaning; preserve punctuation/style information."""
    if not isinstance(text, str):
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def raid_label(model):
    return 0 if model == "human" else 1


def collect_raid_sample(sample_size=500_000, seed=42):
    """
    Memory-safe fallback sampler for the RAID HF training stream.

    The local 500K CSV is preferred. This function is only used if it is
    missing. It balances the eight main English RAID domains by Human/AI.
    """
    from datasets import load_dataset

    target = sample_size // (len(EXPECTED_DOMAINS) * 2)
    buckets = {
        (domain, label): []
        for domain in EXPECTED_DOMAINS
        for label in [0, 1]
    }

    stream = load_dataset(
        "liamdugan/raid",
        "raid",
        split="train",
        streaming=True,
    )

    scanned = 0
    for record in stream:
        scanned += 1
        domain = record.get("domain")
        model = record.get("model")
        text = record.get("generation")

        if domain not in EXPECTED_DOMAINS:
            continue
        if not isinstance(text, str):
            continue

        label = raid_label(model)
        bucket = buckets[(domain, label)]

        # Keep a small reservoir inside each bucket.
        if len(bucket) < target:
            bucket.append(record)
        else:
            j = random.randint(0, scanned - 1)
            if j < target:
                bucket[j] = record

        if all(len(v) >= target for v in buckets.values()):
            break

    rows = []
    for values in buckets.values():
        rows.extend(values)

    rng = random.Random(seed)
    rng.shuffle(rows)

    df = pd.DataFrame(rows)
    print("[OK] Fallback RAID stream sample collected.")
    print("[OK] Records scanned:", f"{scanned:,}")
    return df


if INPUT_CSV.exists():
    print("[DATA] Loading local 500K dataset:", INPUT_CSV)
    df = pd.read_csv(INPUT_CSV)
else:
    print("[DATA] Local 500K dataset not found.")
    print("[DATA] Falling back to streaming RAID sampling.")
    df = collect_raid_sample(FALLBACK_SAMPLE_SIZE, SEED)

required_columns = {
    "id", "source_id", "model", "domain", "generation", "attack"
}
missing = required_columns - set(df.columns)
if missing:
    raise ValueError(f"Missing required RAID columns: {sorted(missing)}")

df["label"] = df["model"].apply(raid_label)
df["text"] = df["generation"].apply(clean_text)

before = len(df)
df = df[df["text"].str.len() >= 50].copy()
df = df.drop_duplicates("id").reset_index(drop=True)

print("[OK] Raw rows:", f"{before:,}")
print("[OK] Rows after cleaning:", f"{len(df):,}")
print("[OK] Duplicate IDs:", df["id"].duplicated().sum())
print("[OK] Human:", int((df["label"] == 0).sum()))
print("[OK] AI:", int((df["label"] == 1).sum()))


# ============================================================
# 3. EDA OUTPUTS
# ============================================================

df["char_length"] = df["text"].str.len()
df["word_length"] = df["text"].str.split().str.len()

eda_summary = df[["char_length", "word_length"]].describe()
eda_summary.to_csv(REPORTS_DIR / "eda_text_length_summary.csv")

domain_counts = df["domain"].value_counts().sort_index()
domain_counts.to_csv(REPORTS_DIR / "domain_distribution.csv")

label_counts = (
    df["label"].map({0: "Human", 1: "AI"})
    .value_counts()
    .rename_axis("Class")
    .reset_index(name="Count")
)
label_counts.to_csv(REPORTS_DIR / "label_distribution.csv", index=False)

print("[OK] EDA summaries saved.")


# ============================================================
# 4. GROUP-AWARE SPLIT
# ============================================================

df["group_id"] = df["source_id"].fillna(df["id"]).astype(str)

splitter_1 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.30,
    random_state=SEED
)
train_idx, temp_idx = next(
    splitter_1.split(df, y=df["label"], groups=df["group_id"])
)

train_df = df.iloc[train_idx].copy()
temp_df = df.iloc[temp_idx].copy()

splitter_2 = GroupShuffleSplit(
    n_splits=1,
    test_size=0.50,
    random_state=SEED
)
val_idx, test_idx = next(
    splitter_2.split(
        temp_df,
        y=temp_df["label"],
        groups=temp_df["group_id"]
    )
)

validation_df = temp_df.iloc[val_idx].copy()
test_df = temp_df.iloc[test_idx].copy()

train_groups = set(train_df["group_id"])
val_groups = set(validation_df["group_id"])
test_groups = set(test_df["group_id"])

print("\n" + "-" * 78)
print("GROUP-AWARE SPLIT")
print("-" * 78)
print("[OK] Train:", f"{len(train_df):,}")
print("[OK] Validation:", f"{len(validation_df):,}")
print("[OK] Test:", f"{len(test_df):,}")
print("[OK] Train ∩ Validation:", len(train_groups & val_groups))
print("[OK] Train ∩ Test:", len(train_groups & test_groups))
print("[OK] Validation ∩ Test:", len(val_groups & test_groups))

if train_groups & val_groups or train_groups & test_groups or val_groups & test_groups:
    raise RuntimeError("Group leakage detected.")

train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
validation_df.to_csv(PROCESSED_DIR / "validation.csv", index=False)
test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

print("[OK] Split files saved.")


# ============================================================
# 5. EVALUATION FUNCTIONS
# ============================================================

X_train = train_df["text"].tolist()
y_train = train_df["label"].to_numpy()

X_val = validation_df["text"].tolist()
y_val = validation_df["label"].to_numpy()

X_test = test_df["text"].tolist()
y_test = test_df["label"].to_numpy()


def evaluate_scores(y_true, y_score, name, threshold=0.5):
    """Return standard metrics from a continuous AI score."""
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    result = {
        "Model": name,
        "Threshold": float(threshold),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_score),
        "FPR": fp / (fp + tn) if (fp + tn) else 0.0,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }

    print(f"\n[{name}]")
    print(
        f"Accuracy={result['Accuracy']:.4f} | "
        f"Precision={result['Precision']:.4f} | "
        f"Recall={result['Recall']:.4f} | "
        f"F1={result['F1']:.4f} | "
        f"ROC-AUC={result['ROC-AUC']:.4f} | "
        f"FPR={result['FPR']:.4f}"
    )
    print("Confusion matrix:")
    print(np.array([[tn, fp], [fn, tp]]))
    return result


def print_step(message):
    print("\n" + "=" * 78)
    print(message)
    print("=" * 78)


# ============================================================
# 6. MODEL 1 — LOGISTIC REGRESSION
# ============================================================

print_step("MODEL 1 — TF-IDF + LOGISTIC REGRESSION")

lr_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.98,
        sublinear_tf=True,
        max_features=250_000,
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
    )),
])

t0 = time.time()
lr_model.fit(X_train, y_train)
lr_train_seconds = time.time() - t0

lr_val_score = lr_model.predict_proba(X_val)[:, 1]
lr_test_score = lr_model.predict_proba(X_test)[:, 1]

lr_val_metrics = evaluate_scores(
    y_val, lr_val_score, "TF-IDF + Logistic Regression — Validation"
)
lr_test_metrics = evaluate_scores(
    y_test, lr_test_score, "TF-IDF + Logistic Regression — Test"
)

print("[OK] Logistic Regression complete.")
print(f"[OK] Training time: {lr_train_seconds:.2f} seconds.")


# ============================================================
# 7. MODEL 2 — CALIBRATED LINEAR SVM
# ============================================================

print_step("MODEL 2 — TF-IDF + CALIBRATED LINEAR SVM")

svm_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.98,
        sublinear_tf=True,
        max_features=250_000,
    )),
    ("classifier", CalibratedClassifierCV(
        LinearSVC(
            class_weight="balanced",
            random_state=SEED,
        ),
        cv=3,
    )),
])

t0 = time.time()
svm_model.fit(X_train, y_train)
svm_train_seconds = time.time() - t0

svm_val_score = svm_model.predict_proba(X_val)[:, 1]
svm_test_score = svm_model.predict_proba(X_test)[:, 1]

svm_val_metrics = evaluate_scores(
    y_val, svm_val_score, "TF-IDF + Calibrated Linear SVM — Validation"
)
svm_test_metrics = evaluate_scores(
    y_test, svm_test_score, "TF-IDF + Calibrated Linear SVM — Test"
)

print("[OK] SVM complete.")
print(f"[OK] Training time: {svm_train_seconds:.2f} seconds.")


# ============================================================
# 8. MODEL 3 — XGBOOST + LINGUISTIC FEATURES
# ============================================================

print_step("MODEL 3 — XGBOOST + LINGUISTIC FEATURES")


def linguistic_features(text):
    text = text if isinstance(text, str) else ""
    words = re.findall(r"\b\w+\b", text)
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]

    n_words = len(words)
    n_sent = max(1, len(sentences))
    n_chars = len(text)
    unique = len(set(w.lower() for w in words))

    return {
        "char_count": n_chars,
        "word_count": n_words,
        "sentence_count": n_sent,
        "avg_word_length": (
            np.mean([len(w) for w in words]) if words else 0
        ),
        "avg_sentence_length": n_words / n_sent,
        "unique_word_ratio": unique / n_words if n_words else 0,
        "digit_ratio": (
            sum(c.isdigit() for c in text) / n_chars
            if n_chars else 0
        ),
        "uppercase_ratio": (
            sum(c.isupper() for c in text) / n_chars
            if n_chars else 0
        ),
        "punctuation_ratio": (
            sum(
                not c.isalnum() and not c.isspace()
                for c in text
            ) / n_chars
            if n_chars else 0
        ),
        "question_count": text.count("?"),
        "exclamation_count": text.count("!"),
        "comma_count": text.count(","),
        "semicolon_count": text.count(";"),
        "colon_count": text.count(":"),
        "newline_count": text.count("\n"),
    }


def make_linguistic_features(texts):
    return pd.DataFrame(
        [linguistic_features(text) for text in texts]
    )


X_train_ling = make_linguistic_features(X_train)
X_val_ling = make_linguistic_features(X_val)
X_test_ling = make_linguistic_features(X_test)

xgb_model = None
xgb_val_score = None
xgb_test_score = None
xgb_val_metrics = None
xgb_test_metrics = None

if XGB_AVAILABLE:
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=SEED,
        n_jobs=-1,
    )

    t0 = time.time()
    xgb_model.fit(
        X_train_ling,
        y_train,
        eval_set=[(X_val_ling, y_val)],
        verbose=False,
    )
    xgb_train_seconds = time.time() - t0

    xgb_val_score = xgb_model.predict_proba(
        X_val_ling
    )[:, 1]
    xgb_test_score = xgb_model.predict_proba(
        X_test_ling
    )[:, 1]

    xgb_val_metrics = evaluate_scores(
        y_val,
        xgb_val_score,
        "XGBoost + Linguistic Features — Validation",
    )
    xgb_test_metrics = evaluate_scores(
        y_test,
        xgb_test_score,
        "XGBoost + Linguistic Features — Test",
    )

    print("[OK] XGBoost complete.")
    print(f"[OK] Training time: {xgb_train_seconds:.2f} seconds.")
else:
    print("[SKIP] XGBoost is unavailable.")


# ============================================================
# 9. MODEL 4 — DISTILROBERTA
# ============================================================

print_step("MODEL 4 — DISTILROBERTA")

transformer_available = False
transformer_val_score = None
transformer_test_score = None
transformer_val_metrics = None
transformer_test_metrics = None
trainer = None
tokenizer = None

if TORCH_AVAILABLE:
    try:
        from datasets import Dataset
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            TrainingArguments,
            Trainer,
        )
        from scipy.special import softmax

        transformer_available = True
    except Exception as exc:
        print("[WARNING] Transformer imports unavailable:", exc)

if transformer_available:
    tokenizer = AutoTokenizer.from_pretrained(
        TRANSFORMER_MODEL
    )

    transformer_train_df = (
        train_df
        .groupby("label", group_keys=False)
        .apply(
            lambda part: part.sample(
                n=min(
                    len(part),
                    TRANSFORMER_TRAIN_LIMIT // 2
                ),
                random_state=SEED,
            )
        )
        .reset_index(drop=True)
    )

    print(
        "[OK] Transformer training subset:",
        f"{len(transformer_train_df):,}"
    )

    train_hf = Dataset.from_pandas(
        transformer_train_df[["text", "label"]],
        preserve_index=False,
    )

    val_hf = Dataset.from_pandas(
        validation_df[["text", "label"]],
        preserve_index=False,
    )

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_hf = train_hf.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )

    val_hf = val_hf.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )

    transformer_model = AutoModelForSequenceClassification.from_pretrained(
        TRANSFORMER_MODEL,
        num_labels=2,
    )

    # This classifier head is newly created for the Human/AI task.
    # Its initialization warning is expected when loading the base model.
    if torch.cuda.is_available():
        transformer_model.gradient_checkpointing_enable()
        print("[OK] Gradient checkpointing enabled.")

    def hf_metrics(eval_pred):
        logits, labels = eval_pred
        pred = np.argmax(logits, axis=-1)
        p, r, f, _ = __import__(
            "sklearn.metrics",
            fromlist=["precision_recall_fscore_support"]
        ).precision_recall_fscore_support(
            labels,
            pred,
            average="binary",
            zero_division=0,
        )
        return {
            "accuracy": accuracy_score(labels, pred),
            "precision": p,
            "recall": r,
            "f1": f,
        }

    training_args = TrainingArguments(
        output_dir=str(
            MODELS_DIR / "distilroberta_checkpoints"
        ),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=TRANSFORMER_LR,
        per_device_train_batch_size=TRANSFORMER_BATCH_SIZE,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=TRANSFORMER_GRAD_ACCUMULATION,
        num_train_epochs=TRANSFORMER_EPOCHS,
        weight_decay=0.01,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
    )

    trainer = Trainer(
        model=transformer_model,
        args=training_args,
        train_dataset=train_hf,
        eval_dataset=val_hf,
        compute_metrics=hf_metrics,
    )

    t0 = time.time()
    trainer.train()
    transformer_train_seconds = time.time() - t0

    val_output = trainer.predict(val_hf)
    transformer_val_score = softmax(
        val_output.predictions,
        axis=1,
    )[:, 1]

    transformer_val_metrics = evaluate_scores(
        y_val,
        transformer_val_score,
        "DistilRoBERTa — Validation",
    )

    print("[OK] DistilRoBERTa training/evaluation complete.")
    print(
        f"[OK] Training time: "
        f"{transformer_train_seconds:.2f} seconds."
    )
else:
    print("[SKIP] DistilRoBERTa unavailable.")


# ============================================================
# 10. MODEL COMPARISON
# ============================================================

print_step("VALIDATION MODEL COMPARISON")

validation_metrics = [
    lr_val_metrics,
    svm_val_metrics,
]

score_map = {
    "LR": lr_val_score,
    "SVM": svm_val_score,
}

if xgb_val_metrics is not None:
    validation_metrics.append(xgb_val_metrics)
    score_map["XGBoost"] = xgb_val_score

if transformer_val_metrics is not None:
    validation_metrics.append(transformer_val_metrics)
    score_map["DistilRoBERTa"] = transformer_val_score

validation_results = pd.DataFrame(validation_metrics)
validation_results.to_csv(
    REPORTS_DIR / "validation_model_results.csv",
    index=False,
)

print(validation_results.sort_values("F1", ascending=False))


# ============================================================
# 11. VALIDATION-ONLY ENSEMBLE WEIGHT + THRESHOLD SEARCH
# ============================================================

print_step("VALIDATION-ONLY ENSEMBLE SEARCH")

model_names = list(score_map.keys())
score_matrix = np.column_stack(
    [score_map[name] for name in model_names]
)

# Deterministic candidate set.
rng = np.random.default_rng(SEED)

candidate_weights = [
    np.eye(len(model_names))[i]
    for i in range(len(model_names))
]

candidate_weights += [
    rng.dirichlet(
        np.ones(len(model_names)
    ))
    for _ in range(5000)
]

thresholds = np.round(
    np.arange(0.40, 0.701, 0.01),
    2,
)

best = {
    "F1": -1,
    "Accuracy": -1,
    "weights": None,
    "threshold": None,
    "score": None,
}

for weights in candidate_weights:
    combined = score_matrix @ weights

    for threshold in thresholds:
        pred = (combined >= threshold).astype(int)
        f1 = f1_score(
            y_val,
            pred,
            zero_division=0,
        )

        if f1 > best["F1"]:
            best["F1"] = float(f1)
            best["weights"] = weights.copy()
            best["threshold"] = float(threshold)
            best["score"] = combined.copy()

best_weights = best["weights"]
best_threshold = best["threshold"]
ensemble_val_score = best["score"]

ensemble_val_metrics = evaluate_scores(
    y_val,
    ensemble_val_score,
    "TextTrace Ensemble — Validation",
    threshold=best_threshold,
)

print("[OK] Validation-only ensemble search complete.")
print("[OK] Selected threshold:", best_threshold)
for name, weight in zip(model_names, best_weights):
    print(f"[OK] {name:18s}: {weight:.4f}")


# ============================================================
# 12. FINAL HELD-OUT TEST
# ============================================================

print_step("FINAL HELD-OUT TEST")

test_score_map = {
    "LR": lr_test_score,
    "SVM": svm_test_score,
}

if xgb_test_score is not None:
    test_score_map["XGBoost"] = xgb_test_score

if transformer_available:
    test_hf = Dataset.from_pandas(
        test_df[["text", "label"]],
        preserve_index=False,
    )

    test_hf = test_hf.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )

    test_output = trainer.predict(test_hf)
    transformer_test_score = softmax(
        test_output.predictions,
        axis=1,
    )[:, 1]

    transformer_test_metrics = evaluate_scores(
        y_test,
        transformer_test_score,
        "DistilRoBERTa — Test",
    )

    test_score_map["DistilRoBERTa"] = transformer_test_score

test_matrix = np.column_stack(
    [test_score_map[name] for name in model_names]
)

ensemble_test_score = test_matrix @ best_weights

ensemble_test_metrics = evaluate_scores(
    y_test,
    ensemble_test_score,
    "TextTrace Ensemble — Final Test",
    threshold=best_threshold,
)

print("[OK] Final held-out test completed.")
print("[IMPORTANT] Test set was not used to choose the ensemble weights.")


# ============================================================
# 13. FINAL MODEL RESULTS TABLE
# ============================================================

final_metrics = [
    lr_test_metrics,
    svm_test_metrics,
]

if xgb_test_metrics is not None:
    final_metrics.append(xgb_test_metrics)

if transformer_test_metrics is not None:
    final_metrics.append(transformer_test_metrics)

final_metrics.append(ensemble_test_metrics)

final_results_df = (
    pd.DataFrame(final_metrics)
    .sort_values("F1", ascending=False)
    .reset_index(drop=True)
)

final_results_df.to_csv(
    REPORTS_DIR / "final_model_results.csv",
    index=False,
)

print_step("FINAL TEST RESULTS")
print(final_results_df.to_string(index=False))


# ============================================================
# 14. CONFUSION MATRIX + ROC CURVES
# ============================================================

def save_confusion_matrix(y_true, score, threshold, filename, title):
    pred = (score >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    ax.imshow(cm)

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Human", "AI"])
    ax.set_yticklabels(["Human", "AI"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(CHARTS_DIR / filename, dpi=220)
    plt.close(fig)


save_confusion_matrix(
    y_test,
    ensemble_test_score,
    best_threshold,
    "01_final_ensemble_confusion_matrix.png",
    "TextTrace Final Ensemble — Confusion Matrix",
)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)

for name in model_names:
    fpr, tpr, _ = roc_curve(
        y_test,
        test_score_map[name],
    )
    auc = roc_auc_score(
        y_test,
        test_score_map[name],
    )
    ax.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc:.3f})",
    )

fpr, tpr, _ = roc_curve(
    y_test,
    ensemble_test_score,
)
auc = roc_auc_score(
    y_test,
    ensemble_test_score,
)
ax.plot(
    fpr,
    tpr,
    label=f"Ensemble (AUC={auc:.3f})",
    linewidth=2.5,
)

ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("TextTrace ROC Curves — Held-Out Test")
ax.legend()
fig.tight_layout()
fig.savefig(
    CHARTS_DIR / "02_roc_curves_test.png",
    dpi=220,
)
plt.close(fig)

# F1 comparison.
plot_df = final_results_df.copy()
fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)
ax.bar(plot_df["Model"], plot_df["F1"])
ax.set_ylabel("F1")
ax.set_title("TextTrace Model Comparison — F1")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=35)
fig.tight_layout()
fig.savefig(
    CHARTS_DIR / "03_model_f1_comparison.png",
    dpi=220,
)
plt.close(fig)

# Accuracy/precision/recall.
metric_cols = ["Accuracy", "Precision", "Recall", "F1"]
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
x = np.arange(len(plot_df))
width = 0.18

for i, metric in enumerate(metric_cols):
    ax.bar(
        x + (i - 1.5) * width,
        plot_df[metric],
        width,
        label=metric,
    )

ax.set_xticks(x)
ax.set_xticklabels(
    plot_df["Model"],
    rotation=35,
    ha="right",
)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("TextTrace Test Metrics")
ax.legend()
fig.tight_layout()
fig.savefig(
    CHARTS_DIR / "04_test_metrics_comparison.png",
    dpi=220,
)
plt.close(fig)

print("[OK] Core evaluation charts saved.")


# ============================================================
# 15. XAI — LOGISTIC REGRESSION
# ============================================================

def explain_lr_text(text, pipeline, top_n=15):
    vec = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["classifier"]

    x = vec.transform([text])
    names = np.array(vec.get_feature_names_out())
    idx = x.nonzero()[1]

    contributions = (
        x[0, idx].toarray().ravel()
        * clf.coef_[0, idx]
    )

    order = np.argsort(
        np.abs(contributions)
    )[::-1][:top_n]

    return pd.DataFrame({
        "feature": names[idx][order],
        "contribution": contributions[order],
    })


xai_df = explain_lr_text(
    test_df.iloc[0]["text"],
    lr_model,
    top_n=15,
)

xai_df.to_csv(
    REPORTS_DIR / "lr_xai_example.csv",
    index=False,
)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
xai_plot = xai_df.sort_values("contribution")
ax.barh(
    xai_plot["feature"],
    xai_plot["contribution"],
)
ax.axvline(0, linestyle="--")
ax.set_xlabel("Contribution to AI direction (+) / Human direction (-)")
ax.set_title("Example Logistic Regression Explanation")
fig.tight_layout()
fig.savefig(
    CHARTS_DIR / "05_lr_xai_contributions.png",
    dpi=220,
)
plt.close(fig)

print("[OK] Logistic Regression XAI output saved.")


# ============================================================
# 16. XGBOOST FEATURE IMPORTANCE
# ============================================================

if xgb_model is not None:
    xgb_importance = pd.DataFrame({
        "feature": X_train_ling.columns,
        "importance": xgb_model.feature_importances_,
    }).sort_values(
        "importance",
        ascending=False,
    )

    xgb_importance.to_csv(
        REPORTS_DIR / "xgb_feature_importance.csv",
        index=False,
    )

    top_xgb = xgb_importance.head(15).sort_values(
        "importance"
    )

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.barh(
        top_xgb["feature"],
        top_xgb["importance"],
    )
    ax.set_xlabel("Feature importance")
    ax.set_title("Top XGBoost Linguistic Features")
    fig.tight_layout()
    fig.savefig(
        CHARTS_DIR / "06_xgb_feature_importance.png",
        dpi=220,
    )
    plt.close(fig)

    print("[OK] XGBoost feature importance saved.")


# ============================================================
# 17. ERROR ANALYSIS
# ============================================================

error_df = test_df[
    ["id", "domain", "model", "attack", "text", "label"]
].copy()

error_df["ai_score"] = ensemble_test_score
error_df["prediction"] = (
    ensemble_test_score >= best_threshold
).astype(int)

error_df["error_type"] = "Correct"

error_df.loc[
    (error_df["label"] == 0)
    & (error_df["prediction"] == 1),
    "error_type",
] = "False Positive"

error_df.loc[
    (error_df["label"] == 1)
    & (error_df["prediction"] == 0),
    "error_type",
] = "False Negative"

errors_only = error_df[
    error_df["error_type"] != "Correct"
].copy()

errors_only.to_csv(
    REPORTS_DIR / "error_analysis.csv",
    index=False,
)

fp_by_domain = (
    errors_only[
        errors_only["error_type"] == "False Positive"
    ]["domain"]
    .value_counts()
    .sort_values(ascending=True)
)

fn_by_domain = (
    errors_only[
        errors_only["error_type"] == "False Negative"
    ]["domain"]
    .value_counts()
    .sort_values(ascending=True)
)

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111)
domains = sorted(
    set(fp_by_domain.index) | set(fn_by_domain.index)
)
fp_values = [
    fp_by_domain.get(d, 0)
    for d in domains
]
fn_values = [
    fn_by_domain.get(d, 0)
    for d in domains
]

x = np.arange(len(domains))
width = 0.38
ax.bar(x - width / 2, fp_values, width, label="False Positive")
ax.bar(x + width / 2, fn_values, width, label="False Negative")
ax.set_xticks(x)
ax.set_xticklabels(domains, rotation=35, ha="right")
ax.set_ylabel("Number of errors")
ax.set_title("TextTrace Error Analysis by Domain")
ax.legend()
fig.tight_layout()
fig.savefig(
    CHARTS_DIR / "07_error_analysis_by_domain.png",
    dpi=220,
)
plt.close(fig)

print("[OK] Error analysis saved.")
print("[OK] False positives:", int(
    (error_df["error_type"] == "False Positive").sum()
))
print("[OK] False negatives:", int(
    (error_df["error_type"] == "False Negative").sum()
))


# ============================================================
# 18. ATTACK / ROBUSTNESS ANALYSIS
# ============================================================

attack_df = error_df.copy()
attack_df["attack"] = attack_df["attack"].fillna("none")

attack_rows = []

for attack, group in attack_df.groupby("attack"):
    if len(group) < 2:
        continue

    attack_rows.append({
        "Attack": attack,
        "N": len(group),
        "Accuracy": accuracy_score(
            group["label"],
            group["prediction"],
        ),
        "Precision": precision_score(
            group["label"],
            group["prediction"],
            zero_division=0,
        ),
        "Recall": recall_score(
            group["label"],
            group["prediction"],
            zero_division=0,
        ),
        "F1": f1_score(
            group["label"],
            group["prediction"],
            zero_division=0,
        ),
    })

attack_results = pd.DataFrame(
    attack_rows
).sort_values(
    "F1",
    ascending=False,
)

attack_results.to_csv(
    REPORTS_DIR / "attack_results.csv",
    index=False,
)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
attack_plot = attack_results.sort_values("F1")
ax.barh(
    attack_plot["Attack"],
    attack_plot["F1"],
)
ax.set_xlabel("F1")
ax.set_title("TextTrace Robustness — F1 by RAID Attack")
ax.set_xlim(0, 1)
fig.tight_layout()
fig.savefig(
    CHARTS_DIR / "08_attack_robustness_f1.png",
    dpi=220,
)
plt.close(fig)

print("[OK] Attack robustness analysis saved.")


# ============================================================
# 19. ADAPTIVE / HARD-EXAMPLE EXPERIMENT
# ============================================================

print_step("ADAPTIVE / HARD-EXAMPLE EXPERIMENT")

initial_train_score = lr_model.predict_proba(
    X_train
)[:, 1]

initial_train_pred = (
    initial_train_score >= 0.5
).astype(int)

hard_mask = (
    (initial_train_pred != y_train)
    | (np.abs(initial_train_score - 0.5) < 0.10)
)

sample_weight = np.ones(
    len(train_df),
    dtype=float,
)
sample_weight[hard_mask] = 2.0

adaptive_lr = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.98,
        sublinear_tf=True,
        max_features=250_000,
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
    )),
])

adaptive_lr.fit(
    X_train,
    y_train,
    classifier__sample_weight=sample_weight,
)

adaptive_val_score = adaptive_lr.predict_proba(
    X_val
)[:, 1]

adaptive_val_metrics = evaluate_scores(
    y_val,
    adaptive_val_score,
    "Adaptive Hard-Example Logistic Regression — Validation",
)

adaptive_comparison = pd.DataFrame([
    {
        "Stage": "Initial Logistic Regression",
        "F1": lr_val_metrics["F1"],
        "Accuracy": lr_val_metrics["Accuracy"],
        "Precision": lr_val_metrics["Precision"],
        "Recall": lr_val_metrics["Recall"],
        "ROC-AUC": lr_val_metrics["ROC-AUC"],
        "FPR": lr_val_metrics["FPR"],
    },
    {
        "Stage": "Hard-example weighted Logistic Regression",
        "F1": adaptive_val_metrics["F1"],
        "Accuracy": adaptive_val_metrics["Accuracy"],
        "Precision": adaptive_val_metrics["Precision"],
        "Recall": adaptive_val_metrics["Recall"],
        "ROC-AUC": adaptive_val_metrics["ROC-AUC"],
        "FPR": adaptive_val_metrics["FPR"],
    },
])

adaptive_comparison.to_csv(
    REPORTS_DIR / "adaptive_comparison.csv",
    index=False,
)

fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)
ax.bar(
    adaptive_comparison["Stage"],
    adaptive_comparison["F1"],
)
ax.set_ylabel("Validation F1")
ax.set_title("Adaptive Hard-Example Experiment")
ax.set_ylim(0, 1)
ax.tick_params(axis="x", rotation=20)
fig.tight_layout()
fig.savefig(
    CHARTS_DIR / "09_adaptive_learning_comparison.png",
    dpi=220,
)
plt.close(fig)

print(
    "[OK] Hard/uncertain training examples:",
    int(hard_mask.sum())
)


# ============================================================
# 20. STREAMLIT-READY DOCUMENT-LEVEL SCORING FUNCTION
# ============================================================

def score_document_from_ensemble(
    text,
    lr_pipeline,
    svm_pipeline,
    xgb_classifier,
    transformer_trainer=None,
    transformer_tokenizer=None,
    weights=None,
    threshold=0.5,
):
    """
    Return the ML-stage result that Streamlit can later display.

    IMPORTANT:
    - ai_score is an AI-likelihood model score, not proof of authorship.
    - similarity_percentage is NOT produced here.
      It belongs to the separate similarity engine in Streamlit.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Document text is empty.")

    scores = []

    lr_score = lr_pipeline.predict_proba([text])[0, 1]
    svm_score = svm_pipeline.predict_proba([text])[0, 1]

    scores.extend([
        ("LR", float(lr_score)),
        ("SVM", float(svm_score)),
    ])

    if xgb_classifier is not None:
        xgb_features = make_linguistic_features([text])
        xgb_score = xgb_classifier.predict_proba(
            xgb_features
        )[0, 1]
        scores.append(("XGBoost", float(xgb_score)))

    if (
        transformer_trainer is not None
        and transformer_tokenizer is not None
    ):
        from datasets import Dataset
        from scipy.special import softmax

        one = Dataset.from_dict({
            "text": [text],
            "label": [0],
        })

        one = one.map(
            lambda batch: transformer_tokenizer(
                batch["text"],
                truncation=True,
                max_length=MAX_LENGTH,
            ),
            batched=True,
            remove_columns=["text"],
        )

        output = transformer_trainer.predict(one)
        score = softmax(
            output.predictions,
            axis=1,
        )[0, 1]

        scores.append(
            ("DistilRoBERTa", float(score))
        )

    score_dict = dict(scores)

    if weights is None:
        weights = {
            name: 1 / len(scores)
            for name, _ in scores
        }

    common_names = [
        name for name, _ in scores
        if name in weights
    ]

    total_weight = sum(
        weights[name]
        for name in common_names
    )

    ai_score = sum(
        score_dict[name] * weights[name]
        for name in common_names
    ) / total_weight

    label = "AI-generated" if ai_score >= threshold else "Human-written"

    distance = abs(ai_score - threshold)
    if distance < 0.05:
        confidence = "Uncertain"
    elif distance < 0.15:
        confidence = "Moderate"
    else:
        confidence = "Higher"

    return {
        "ai_score": float(ai_score),
        "ai_percentage": float(ai_score * 100),
        "label": label,
        "confidence": confidence,
        "threshold": float(threshold),
        "model_scores": score_dict,
        "similarity_percentage": None,
        "similarity_status": "Run separate similarity engine in Streamlit.",
    }


# ============================================================
# 21. SAVE MODELS
# ============================================================

print_step("SAVING MODELS AND REPRODUCIBILITY FILES")

joblib.dump(
    lr_model,
    MODELS_DIR / "tfidf_logistic_regression.pkl",
)

joblib.dump(
    svm_model,
    MODELS_DIR / "tfidf_calibrated_linear_svm.pkl",
)

joblib.dump(
    adaptive_lr,
    MODELS_DIR / "adaptive_logistic_regression.pkl",
)

if xgb_model is not None:
    joblib.dump(
        xgb_model,
        MODELS_DIR / "xgboost_linguistic.pkl",
    )

if transformer_available and trainer is not None:
    transformer_dir = (
        MODELS_DIR / "distilroberta_texttrace"
    )
    transformer_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(str(transformer_dir))
    tokenizer.save_pretrained(
        str(transformer_dir)
    )

    print("[OK] DistilRoBERTa saved.")

ensemble_config = {
    "model_names": model_names,
    "weights": {
        name: float(weight)
        for name, weight in zip(
            model_names,
            best_weights,
        )
    },
    "threshold": float(best_threshold),
    "label_mapping": {
        "0": "Human",
        "1": "AI",
    },
    "seed": SEED,
    "dataset": "liamdugan/raid",
    "local_dataset": str(INPUT_CSV),
    "transformer_model": TRANSFORMER_MODEL,
    "transformer_train_limit": TRANSFORMER_TRAIN_LIMIT,
}

with open(
    MODELS_DIR / "ensemble_config.json",
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        ensemble_config,
        handle,
        indent=2,
    )

metadata = {
    "project": "TextTrace",
    "purpose": "AI-generated text detection",
    "dataset": "RAID",
    "dataset_configuration": "raid",
    "rows_after_cleaning": int(len(df)),
    "train_rows": int(len(train_df)),
    "validation_rows": int(len(validation_df)),
    "test_rows": int(len(test_df)),
    "seed": SEED,
    "domains": EXPECTED_DOMAINS,
    "models": model_names,
    "ensemble_threshold": float(best_threshold),
    "ensemble_weights": {
        name: float(weight)
        for name, weight in zip(
            model_names,
            best_weights,
        )
    },
    "note": (
        "AI score and source similarity are separate outputs. "
        "AI score must not be described as plagiarism proof."
    ),
}

with open(
    REPORTS_DIR / "reproducibility_metadata.json",
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        metadata,
        handle,
        indent=2,
    )

print("[OK] Models saved.")
print("[OK] Ensemble configuration saved.")
print("[OK] Reproducibility metadata saved.")


# ============================================================
# 22. FINAL SUMMARY
# ============================================================

print_step("TEXTTRACE ML STAGE COMPLETE")

print("Final test ensemble:")
for key in [
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC-AUC",
    "FPR",
]:
    print(
        f"  {key:10s}: "
        f"{ensemble_test_metrics[key]:.4f}"
    )

print("\nSelected threshold:", best_threshold)
print("Selected weights:")
for name, weight in zip(
    model_names,
    best_weights,
):
    print(f"  {name:18s}: {weight:.4f}")

print("\nSaved outputs:")
for path in sorted(REPORTS_DIR.glob("*")):
    if path.is_file():
        print("  [OK]", path)

print("\nSaved charts:")
for path in sorted(CHARTS_DIR.glob("*.png")):
    print("  [OK]", path)

print("\nNext stage:")
print("  1. Build Streamlit document parser.")
print("  2. Add TXT/DOCX/PDF support.")
print("  3. Add sentence/paragraph/page location tracking.")
print("  4. Add separate public-web/local-repository similarity engine.")
print("  5. Add report generation.")
print("  6. Never label AI score as plagiarism proof.")
print("\nDONE.")
