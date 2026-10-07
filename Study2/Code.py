import os
import warnings
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import wilcoxon
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss, confusion_matrix,
    precision_score, recall_score, f1_score,
)
from sklearn.model_selection import (
    train_test_split, StratifiedGroupKFold,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from transformers import (
    BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer,
)


SEEDS = (42, 7, 123, 999, 2023, 8888, 7777)

EXCEL_PATH = "/kaggle/input/breast-lesions-usg/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
IMG_DIR    = "/kaggle/input/breast-lesions-usg/images/BrEaST-Lesions_USG-images_and_masks"
SHEET_NAME = "BrEaST-Lesions-USG clinical dat"

OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PER_SEED_CSV = (
    f"{OUTPUT_DIR}/study2_patient_level_no_diagnosis_results.csv"
)

EPOCHS = 5
BATCH_SIZE = 16
MAX_LENGTH = 128
LR = 2e-5
WEIGHT_DECAY = 0.01

TEXT_FEATURE_COLUMNS = [
    "Symptoms",
    "Breast_composition",
    "BIRADS_category",
    "Lesion_shape",
    "Lesion_margin",
    "Lesion_echogenicity",
    "Posterior_features",
    "Calcifications",
]
COLUMN_RENAMES = {
    "CaseID": "Patient_ID",
    "Tissue_composition": "Breast_composition",
    "BIRADS": "BIRADS_category",
    "Shape": "Lesion_shape",
    "Margin": "Lesion_margin",
    "Echogenicity": "Lesion_echogenicity",
}


def build_clinical_text(row):
    parts = [
        f"{c}: {str(row[c]).strip()}"
        for c in TEXT_FEATURE_COLUMNS
        if pd.notna(row[c]) and str(row[c]).strip()
    ]
    return " ".join(parts)


def patient_level_stratified_split(df, seed):
    patients = (
        df[[PATIENT_ID_COLUMN, "Classification"]]
        .drop_duplicates(subset=[PATIENT_ID_COLUMN])
        .reset_index(drop=True)
    )
    train_patients, val_patients = train_test_split(
        patients,
        test_size=0.2,
        random_state=seed,
        stratify=patients["Classification"],
    )
    train_ids = set(train_patients[PATIENT_ID_COLUMN])
    val_ids = set(val_patients[PATIENT_ID_COLUMN])
    return (
        df[df[PATIENT_ID_COLUMN].isin(train_ids)].copy(),
        df[df[PATIENT_ID_COLUMN].isin(val_ids)].copy(),
    )


def seeded_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


PROPOSED = "Multimodal"
MODELS_TO_RUN = ("BERT-only", "GPT2-only", "ResNet18-only", "Multimodal")

PATIENT_ID_COLUMN = "Patient_ID"

CV_FOLDS = 5
CV_SEED = 42


def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (y_prob >= lo) & (y_prob <= hi if i == n_bins - 1
                                   else y_prob < hi)
        if in_bin.sum() == 0:
            continue
        bin_acc = y_true[in_bin].mean()
        bin_conf = y_prob[in_bin].mean()
        ece += (in_bin.sum() / len(y_prob)) * abs(bin_acc - bin_conf)
    return ece


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class TextOnlyDataset(Dataset):

    def __init__(self, df, bert_tok, gpt_tok, max_length=128):
        self.df = df.reset_index(drop=True)
        self.bert_tok = bert_tok
        self.gpt_tok = gpt_tok
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = build_clinical_text(row)

        b = self.bert_tok.encode_plus(
            text, add_special_tokens=True, max_length=self.max_length,
            return_token_type_ids=False, padding="max_length",
            return_attention_mask=True, return_tensors="pt", truncation=True,
        )
        g = self.gpt_tok.encode_plus(
            text, add_special_tokens=True, max_length=self.max_length,
            padding="max_length", return_attention_mask=True,
            return_tensors="pt", truncation=True,
        )
        label = 1 if row["Classification"] == "malignant" else 0
        return {
            "bert_input_ids":      b["input_ids"].squeeze(0),
            "bert_attention_mask": b["attention_mask"].squeeze(0),
            "gpt_input_ids":       g["input_ids"].squeeze(0),
            "gpt_attention_mask":  g["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


class ImageOnlyDataset(Dataset):

    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["Image_filename"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = 1 if row["Classification"] == "malignant" else 0
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }


class TextImageDataset(Dataset):

    def __init__(self, df, img_dir, bert_tok, gpt_tok, transform,
                 max_length=128):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.bert_tok = bert_tok
        self.gpt_tok = gpt_tok
        self.max_length = max_length
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = build_clinical_text(row)

        b = self.bert_tok.encode_plus(
            text, add_special_tokens=True, max_length=self.max_length,
            return_token_type_ids=False, padding="max_length",
            return_attention_mask=True, return_tensors="pt", truncation=True,
        )
        g = self.gpt_tok.encode_plus(
            text, add_special_tokens=True, max_length=self.max_length,
            padding="max_length", return_attention_mask=True,
            return_tensors="pt", truncation=True,
        )

        img_path = os.path.join(self.img_dir, row["Image_filename"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        label = 1 if row["Classification"] == "malignant" else 0
        return {
            "bert_input_ids":      b["input_ids"].squeeze(0),
            "bert_attention_mask": b["attention_mask"].squeeze(0),
            "gpt_input_ids":       g["input_ids"].squeeze(0),
            "gpt_attention_mask":  g["attention_mask"].squeeze(0),
            "image":               image,
            "label": torch.tensor(label, dtype=torch.long),
        }


class BertOnlyClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.bert.config.hidden_size, num_classes),
        )

    def forward(self, bert_input_ids, bert_attention_mask, **_):
        out = self.bert(input_ids=bert_input_ids,
                        attention_mask=bert_attention_mask)
        return self.classifier(out.pooler_output)


class GPT2OnlyClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.gpt = GPT2Model.from_pretrained("gpt2")
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.gpt.config.n_embd, num_classes),
        )

    def forward(self, gpt_input_ids, gpt_attention_mask, **_):
        out = self.gpt(input_ids=gpt_input_ids,
                       attention_mask=gpt_attention_mask)
        last_token_index = gpt_attention_mask.sum(dim=1) - 1
        pooled = out.last_hidden_state[
            torch.arange(out.last_hidden_state.size(0),
                        device=out.last_hidden_state.device),
            last_token_index,
        ]
        return self.classifier(pooled)


class ResNet18OnlyClassifier(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()
        self.resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.resnet.fc.in_features, num_classes),
        )

    def forward(self, image, **_):
        return self.resnet(image)


class MultimodalClassifier(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.gpt = GPT2Model.from_pretrained("gpt2")
        self.resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 128)
        self.bert_proj = nn.Linear(self.bert.config.hidden_size, 128)
        self.gpt_proj  = nn.Linear(self.gpt.config.n_embd, 128)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 3, num_classes),
        )

    def forward(self, bert_input_ids, bert_attention_mask,
                gpt_input_ids, gpt_attention_mask, image, **_):
        b = self.bert(input_ids=bert_input_ids,
                      attention_mask=bert_attention_mask)
        bf = self.bert_proj(b.pooler_output)
        g = self.gpt(input_ids=gpt_input_ids,
                     attention_mask=gpt_attention_mask)
        last_token_index = gpt_attention_mask.sum(dim=1) - 1
        g_pooled = g.last_hidden_state[
            torch.arange(g.last_hidden_state.size(0),
                        device=g.last_hidden_state.device),
            last_token_index,
        ]
        gf = self.gpt_proj(g_pooled)
        imf = self.resnet(image)
        combined = torch.cat((bf, gf, imf), dim=1)
        return self.classifier(combined)


def run_epoch_eval(model, loader, loss_fn, device, needs_text, needs_image):
    model.eval()
    y_true, y_pred, y_prob, losses = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            kwargs = {}
            if needs_text:
                kwargs["bert_input_ids"]      = batch["bert_input_ids"].to(device)
                kwargs["bert_attention_mask"] = batch["bert_attention_mask"].to(device)
                kwargs["gpt_input_ids"]       = batch["gpt_input_ids"].to(device)
                kwargs["gpt_attention_mask"]  = batch["gpt_attention_mask"].to(device)
            if needs_image:
                kwargs["image"] = batch["image"].to(device)
            logits = model(**kwargs)
            labels = batch["label"].to(device)
            losses.append(loss_fn(logits, labels).item())
            y_prob.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            y_true.extend(labels.cpu().numpy().tolist())
    return (np.array(y_true), np.array(y_pred), np.array(y_prob),
            float(np.mean(losses)))


def train_and_evaluate(model_name, train_loader, val_loader, device):
    if model_name == "BERT-only":
        model = BertOnlyClassifier(num_classes=2)
        needs_text  = True
        needs_image = False
    elif model_name == "GPT2-only":
        model = GPT2OnlyClassifier(num_classes=2)
        needs_text  = True
        needs_image = False
    elif model_name == "ResNet18-only":
        model = ResNet18OnlyClassifier(num_classes=2)
        needs_text  = False
        needs_image = True
    elif model_name == "Multimodal":
        model = MultimodalClassifier(num_classes=2)
        needs_text  = True
        needs_image = True
    else:
        raise ValueError(model_name)

    model = model.to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    history = []
    for ep in range(EPOCHS):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimiser.zero_grad()
            kwargs = {}
            if needs_text:
                kwargs["bert_input_ids"]      = batch["bert_input_ids"].to(device)
                kwargs["bert_attention_mask"] = batch["bert_attention_mask"].to(device)
                kwargs["gpt_input_ids"]       = batch["gpt_input_ids"].to(device)
                kwargs["gpt_attention_mask"]  = batch["gpt_attention_mask"].to(device)
            if needs_image:
                kwargs["image"] = batch["image"].to(device)
            logits = model(**kwargs)
            loss = loss_fn(logits, batch["label"].to(device))
            loss.backward()
            optimiser.step()
            train_losses.append(loss.item())

        y_true, y_pred, y_prob, val_loss = run_epoch_eval(
            model, val_loader, loss_fn, device, needs_text, needs_image)
        history.append({
            "epoch": ep + 1,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "val_accuracy": accuracy_score(y_true, y_pred),
        })
        print(f"    epoch {ep + 1}: train_loss={history[-1]['train_loss']:.4f}  "
              f"val_loss={val_loss:.4f}  "
              f"val_acc={history[-1]['val_accuracy']:.4f}")

    metrics = {
        "accuracy":    accuracy_score(y_true, y_pred),
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "recall":      recall_score(y_true, y_pred, zero_division=0),
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":     roc_auc_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "ece":         expected_calibration_error(y_true, y_prob),
    }

    del model
    torch.cuda.empty_cache()
    return metrics, history


def run_five_fold_cv(df, bert_tok, gpt_tok, device):
    skf = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True,
                               random_state=CV_SEED)
    fold_rows = []
    agg_cm = np.zeros((2, 2), dtype=int)

    for fold_i, (tr_idx, val_idx) in enumerate(
            skf.split(df, df["Classification"], groups=df[PATIENT_ID_COLUMN])):
        fold_seed = CV_SEED + fold_i
        set_seed(fold_seed)
        train_df = df.iloc[tr_idx]
        val_df = df.iloc[val_idx]

        train_ds = TextImageDataset(train_df, IMG_DIR, bert_tok, gpt_tok,
                                    image_transform, MAX_LENGTH)
        val_ds = TextImageDataset(val_df, IMG_DIR, bert_tok, gpt_tok,
                                  image_transform, MAX_LENGTH)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                                  shuffle=True,
                                  generator=seeded_generator(fold_seed))
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = MultimodalClassifier(num_classes=2).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=LR,
                                      weight_decay=WEIGHT_DECAY)
        loss_fn = nn.CrossEntropyLoss()

        for ep in range(EPOCHS):
            model.train()
            train_losses = []
            for batch in train_loader:
                optimiser.zero_grad()
                logits = model(
                    bert_input_ids=batch["bert_input_ids"].to(device),
                    bert_attention_mask=batch["bert_attention_mask"].to(device),
                    gpt_input_ids=batch["gpt_input_ids"].to(device),
                    gpt_attention_mask=batch["gpt_attention_mask"].to(device),
                    image=batch["image"].to(device),
                )
                loss = loss_fn(logits, batch["label"].to(device))
                loss.backward()
                optimiser.step()
                train_losses.append(loss.item())

            y_true, y_pred, _, val_loss = run_epoch_eval(
                model, val_loader, loss_fn, device, True, True)
            epoch_acc = accuracy_score(y_true, y_pred)
            print(f"    Fold {fold_i + 1} epoch {ep + 1}: "
                  f"train_loss={np.mean(train_losses):.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={epoch_acc:.4f}")

        val_acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fp = int(cm[0, 1])
        fn = int(cm[1, 0])
        agg_cm += cm

        print(f"  Fold {fold_i + 1}: val_acc={val_acc:.4f}  "
              f"val_loss={val_loss:.4f}  fp={fp}  fn={fn}")

        fold_rows.append({
            "fold": fold_i + 1,
            "val_accuracy": val_acc,
            "val_loss": val_loss,
            "false_positives": fp,
            "false_negatives": fn,
        })

        del model
        torch.cuda.empty_cache()

    df_cv = pd.DataFrame(fold_rows)
    df_cv.to_csv(f"{OUTPUT_DIR}/table5_5_cv_results.csv", index=False)
    pd.DataFrame(agg_cm, index=["true_benign", "true_malignant"],
                 columns=["pred_benign", "pred_malignant"]).to_csv(
        f"{OUTPUT_DIR}/figure5_10_confusion_matrix.csv")

    print(f"  Mean val_accuracy: {df_cv['val_accuracy'].mean():.4f} "
          f"+/- {df_cv['val_accuracy'].std():.4f}")
    print(f"  Total FP={df_cv['false_positives'].sum()}  "
          f"Total FN={df_cv['false_negatives'].sum()}")
    print("  Aggregated confusion matrix:\n", agg_cm)

    tn, fp, fn, tp = agg_cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    print(f"  Sensitivity={sensitivity:.4f}  Specificity={specificity:.4f}  "
          f"PPV={ppv:.4f}  NPV={npv:.4f}")
    pd.DataFrame([{
        "sensitivity": sensitivity, "specificity": specificity,
        "ppv": ppv, "npv": npv,
    }]).to_csv(f"{OUTPUT_DIR}/section5_7_clinical_metrics.csv", index=False)

    return df_cv, agg_cm


def main():
    print("=" * 70)
    print("Study 2 (USG) multi-seed evaluation with calibration + Wilcoxon")
    print("=" * 70)
    print(f"Seeds: {SEEDS}")
    print(f"Models: {MODELS_TO_RUN}")
    print(f"Epochs: {EPOCHS}, batch size: {BATCH_SIZE}, lr: {LR}")
    print()

    df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)
    df = df.rename(columns=COLUMN_RENAMES)
    df = df.dropna(subset=["Image_filename", "Classification"])
    df["Classification"] = (
        df["Classification"].astype(str).str.strip().str.lower()
    )
    df = df[df["Classification"].isin(["benign", "malignant"])]
    df = df.reset_index(drop=True)
    print(f"Loaded {len(df)} cases ({(df['Classification'] == 'malignant').sum()} "
          f"malignant, {(df['Classification'] == 'benign').sum()} benign)\n")

    bert_tok = BertTokenizer.from_pretrained("bert-base-uncased")
    gpt_tok  = GPT2Tokenizer.from_pretrained("gpt2")
    gpt_tok.pad_token = gpt_tok.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    if os.path.exists(PER_SEED_CSV):
        existing = pd.read_csv(PER_SEED_CSV)
        completed = set(zip(existing["seed"], existing["model"]))
        print(f"Resuming: {len(completed)} (seed, model) combos already done.")
    else:
        existing = None
        completed = set()

    rows_so_far = ([] if existing is None
                   else existing.to_dict(orient="records"))

    for seed in SEEDS:
        print(f"\n=== Seed {seed} ===")
        set_seed(seed)

        train_df, val_df = patient_level_stratified_split(df, seed)

        train_text_ds = TextOnlyDataset(train_df, bert_tok, gpt_tok, MAX_LENGTH)
        val_text_ds   = TextOnlyDataset(val_df,   bert_tok, gpt_tok, MAX_LENGTH)
        train_text_loader = DataLoader(train_text_ds, batch_size=BATCH_SIZE,
                                       shuffle=True,
                                       generator=seeded_generator(seed))
        val_text_loader   = DataLoader(val_text_ds,   batch_size=BATCH_SIZE,
                                       shuffle=False)

        train_img_ds = ImageOnlyDataset(train_df, IMG_DIR, image_transform)
        val_img_ds   = ImageOnlyDataset(val_df,   IMG_DIR, image_transform)
        train_img_loader = DataLoader(train_img_ds, batch_size=BATCH_SIZE,
                                      shuffle=True,
                                      generator=seeded_generator(seed))
        val_img_loader   = DataLoader(val_img_ds,   batch_size=BATCH_SIZE,
                                      shuffle=False)

        train_mm_ds = TextImageDataset(train_df, IMG_DIR, bert_tok, gpt_tok,
                                       image_transform, MAX_LENGTH)
        val_mm_ds   = TextImageDataset(val_df,   IMG_DIR, bert_tok, gpt_tok,
                                       image_transform, MAX_LENGTH)
        train_mm_loader = DataLoader(train_mm_ds, batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     generator=seeded_generator(seed))
        val_mm_loader   = DataLoader(val_mm_ds,   batch_size=BATCH_SIZE,
                                     shuffle=False)

        for model_name in MODELS_TO_RUN:
            if (seed, model_name) in completed:
                print(f"  [skip] {model_name} (already done)")
                continue

            if model_name in ("BERT-only", "GPT2-only"):
                train_loader, val_loader = train_text_loader, val_text_loader
            elif model_name == "ResNet18-only":
                train_loader, val_loader = train_img_loader, val_img_loader
            else:
                train_loader, val_loader = train_mm_loader, val_mm_loader

            print(f"  Training {model_name}...", end=" ", flush=True)
            try:
                metrics, history = train_and_evaluate(
                    model_name, train_loader, val_loader, device,
                )
            except Exception as e:
                print(f"ERROR: {e}")
                continue

            print(f"acc={metrics['accuracy']:.4f}  "
                  f"prec={metrics['precision']:.4f}  "
                  f"rec={metrics['recall']:.4f}  "
                  f"f1={metrics['f1']:.4f}  "
                  f"roc={metrics['roc_auc']:.4f}  "
                  f"brier={metrics['brier_score']:.4f}  "
                  f"ece={metrics['ece']:.4f}")

            row = {"seed": seed, "model": model_name, **metrics}
            rows_so_far.append(row)

            pd.DataFrame(rows_so_far).to_csv(PER_SEED_CSV, index=False)

    df_per_seed = pd.DataFrame(rows_so_far)
    summary_rows = []
    for name in df_per_seed["model"].unique():
        sub = df_per_seed[df_per_seed["model"] == name]
        summary_rows.append({
            "model":          name,
            "accuracy_mean":  sub["accuracy"].mean(),
            "accuracy_std":   sub["accuracy"].std(),
            "roc_auc_mean":   sub["roc_auc"].mean(),
            "roc_auc_std":    sub["roc_auc"].std(),
            "brier_mean":     sub["brier_score"].mean(),
            "brier_std":      sub["brier_score"].std(),
            "ece_mean":       sub["ece"].mean(),
            "ece_std":        sub["ece"].std(),
        })
    df_summary = (pd.DataFrame(summary_rows)
                  .sort_values("accuracy_mean", ascending=False)
                  .reset_index(drop=True))

    print("\n" + "=" * 70)
    print(f"Summary across {len(SEEDS)} seeds")
    print("=" * 70)
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 200):
        print(df_summary.to_string(index=False))
    df_summary.to_csv(f"{OUTPUT_DIR}/study2_summary.csv", index=False)
    print(f"\nSaved {OUTPUT_DIR}/study2_summary.csv")

    print("\n" + "=" * 70)
    print(f"Wilcoxon signed-rank tests ({PROPOSED} vs each baseline)")
    print("alternative='greater' tests whether Multimodal > baseline")
    print("=" * 70)

    proposed_accs = (df_per_seed[df_per_seed["model"] == PROPOSED]
                     .sort_values("seed")["accuracy"].values)

    wilcoxon_rows = []
    for name in MODELS_TO_RUN:
        if name == PROPOSED:
            continue
        baseline_accs = (df_per_seed[df_per_seed["model"] == name]
                         .sort_values("seed")["accuracy"].values)
        if len(baseline_accs) != len(proposed_accs):
            print(f"  [skip] {name}: incomplete seed coverage")
            continue
        diffs = proposed_accs - baseline_accs
        if np.allclose(diffs, 0):
            stat, p, note = np.nan, 1.0, "all paired differences are zero"
        else:
            try:
                stat, p = wilcoxon(proposed_accs, baseline_accs,
                                   alternative="greater",
                                   zero_method="wilcox")
                note = ""
            except ValueError as e:
                stat, p, note = np.nan, np.nan, f"wilcoxon error: {e}"

        wilcoxon_rows.append({
            "baseline":             name,
            "multimodal_mean_acc":  proposed_accs.mean(),
            "baseline_mean_acc":    baseline_accs.mean(),
            "diff":                 proposed_accs.mean() - baseline_accs.mean(),
            "wilcoxon_statistic":   stat,
            "p_value":              p,
            "significant_at_005":   (not np.isnan(p)) and (p < 0.05),
            "note":                 note,
        })

    df_wilcoxon = pd.DataFrame(wilcoxon_rows)
    with pd.option_context("display.float_format", "{:.4f}".format,
                           "display.width", 200):
        print(df_wilcoxon.to_string(index=False))
    df_wilcoxon.to_csv(f"{OUTPUT_DIR}/study2_wilcoxon.csv", index=False)
    print(f"\nSaved {OUTPUT_DIR}/study2_wilcoxon.csv")

    print("\n" + "=" * 70)
    print(f"Five-fold cross-validation (Multimodal model only)")
    print("=" * 70)
    run_five_fold_cv(df, bert_tok, gpt_tok, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
