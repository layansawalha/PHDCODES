import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import wilcoxon
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, GPT2Model
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

project_root = pdf_dir = None
for root, _, files in os.walk("/kaggle/input"):
    if "multimodal.py" in files:
        project_root = os.path.dirname(root)
    if root.rstrip("/").endswith("extracted_pdfs/data"):
        pdf_dir = root

if project_root and project_root not in sys.path:
    sys.path.insert(0, project_root)

from pdf_hybrid.multimodal import extract_images_from_pdfs, MultimodalDataset
from pdf_hybrid.data import load_corpus, labels_sbert_kmeans
from pdf_hybrid.training import set_seed

OUT_DIR    = "/kaggle/working/thesis_outputs"
IMG_CACHE  = "/kaggle/working/pdf_images"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(IMG_CACHE, exist_ok=True)

SEEDS      = (42, 7, 123, 999, 2023, 8888, 7777)
N_CLUSTERS = 5
MAX_LEN    = 512
BATCH_SIZE = 16
MM_EPOCHS  = 5
N_IMAGES   = 4
FUSION_DIM = 256
DROPOUT    = 0.5
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_ece(y_true, y_prob, n_bins=10):
    conf = np.max(y_prob, axis=1)
    pred = np.argmax(y_prob, axis=1)
    acc  = (pred == np.asarray(y_true)).astype(float)
    ece  = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        m = ((conf >= lo) & (conf <= hi)) if i == n_bins - 1 else (
            (conf >= lo) & (conf < hi)
        )
        if m.sum() > 0:
            ece += m.sum() * abs(acc[m].mean() - conf[m].mean())
    return ece / len(y_true)

def compute_brier(y_true, y_prob, n_classes):
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    return float(np.mean(np.sum((y_prob - y_bin) ** 2, axis=1)))

def compute_auc(y_true, y_prob, n_classes):
    try:
        return roc_auc_score(
            label_binarize(y_true, classes=list(range(n_classes))),
            y_prob, average="macro", multi_class="ovr")
    except Exception:
        return float("nan")

def metrics_from_pairs(pairs, n_classes):
    rows = []
    for y_t, y_p, y_s in pairs:
        acc = accuracy_score(y_t, y_p)
        _, _, f1, _ = precision_recall_fscore_support(
            y_t, y_p, average="weighted", zero_division=0)
        rows.append({
            "acc":   acc,
            "f1":    f1,
            "auc":   compute_auc(y_t, y_s, n_classes)  if y_s is not None else np.nan,
            "brier": compute_brier(y_t, y_s, n_classes) if y_s is not None else np.nan,
            "ece":   compute_ece(y_t, y_s)               if y_s is not None else np.nan,
        })
    df = pd.DataFrame(rows)
    out = {}
    for col in df.columns:
        out[f"{col}_mean"] = df[col].mean()
        out[f"{col}_std"]  = df[col].std()
    return out, df

class VisualAttentionPool(nn.Module):
    def __init__(self, in_dim=512):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, seq, mask):
        mask = mask.bool()
        scores = self.attention(seq).squeeze(-1)
        scores = scores.masked_fill(~mask, float('-inf'))

        no_images = ~mask.any(dim=1)
        scores = scores.masked_fill(no_images.unsqueeze(1), 0.0)
        weights = F.softmax(scores, dim=-1)
        weights = weights * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        weights = weights.unsqueeze(-1)
        return (seq * weights).sum(dim=1)


def last_real_token(last_hidden_state, attention_mask):
    positions = torch.arange(
        attention_mask.size(1), device=attention_mask.device
    ).unsqueeze(0)
    last_index = (positions * attention_mask.long()).max(dim=1).values
    batch_index = torch.arange(
        last_hidden_state.size(0), device=last_hidden_state.device
    )
    return last_hidden_state[batch_index, last_index]

def _make_head(in_dim, n_classes, dropout=0.5):
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, n_classes))

class MultimodalBertGptResNet(nn.Module):
    def __init__(self, n_classes, fusion_dim=FUSION_DIM, dropout=DROPOUT):
        super().__init__()
        self.bert  = AutoModel.from_pretrained("bert-base-uncased")
        self.gpt2  = GPT2Model.from_pretrained("gpt2")
        h_bert = self.bert.config.hidden_size
        h_gpt = self.gpt2.config.hidden_size

        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.vision_features = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.vision_features.parameters():
            p.requires_grad = False
        
        self.vis_pool = VisualAttentionPool(512)

        self.bert_proj = nn.Sequential(nn.Linear(h_bert, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        self.gpt_proj  = nn.Sequential(nn.Linear(h_gpt, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        self.vis_proj  = nn.Sequential(nn.Linear(512, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())

        self.text_gate = nn.Linear(fusion_dim * 2, fusion_dim)

        self.mm_gate = nn.Linear(fusion_dim * 2, fusion_dim)
        nn.init.zeros_(self.mm_gate.weight)
        nn.init.constant_(self.mm_gate.bias, -4.0)

        self.classifier = _make_head(fusion_dim, n_classes, dropout)

    def _encode_images(self, images, image_mask):
        B, N, C, H, W = images.shape
        with torch.no_grad():
            feat = self.vision_features(images.reshape(B * N, C, H, W)).reshape(B, N, -1)
        return self.vis_pool(feat, image_mask)

    def forward(self, bert_ids, bert_mask, gpt_ids, gpt_mask, images, image_mask, **_):
        bert_hs = self.bert(input_ids=bert_ids, attention_mask=bert_mask).pooler_output
        gpt_output = self.gpt2(input_ids=gpt_ids, attention_mask=gpt_mask)
        gpt_hs = last_real_token(gpt_output.last_hidden_state, gpt_mask)

        z_bert = self.bert_proj(bert_hs)
        z_gpt  = self.gpt_proj(gpt_hs)

        g_text = torch.sigmoid(self.text_gate(torch.cat([z_bert, z_gpt], dim=1)))
        h_text = z_bert + g_text * z_gpt

        vis_hs = self._encode_images(images, image_mask)
        z_img  = self.vis_proj(vis_hs)

        g_mm = torch.sigmoid(self.mm_gate(torch.cat([h_text, z_img], dim=1)))
        h_fusion = h_text + g_mm * z_img

        return self.classifier(h_fusion)

class AblationBertOnly(nn.Module):
    def __init__(self, n_classes, fusion_dim=FUSION_DIM, dropout=DROPOUT):
        super().__init__()
        self.bert  = AutoModel.from_pretrained("bert-base-uncased")
        h = self.bert.config.hidden_size
        self.proj  = nn.Sequential(nn.Linear(h, fusion_dim),
                                   nn.LayerNorm(fusion_dim), nn.GELU())
        self.classifier = _make_head(fusion_dim, n_classes, dropout)

    def forward(self, bert_ids, bert_mask, **_):
        hs = self.bert(input_ids=bert_ids, attention_mask=bert_mask).pooler_output
        return self.classifier(self.proj(hs))

class AblationGpt2Only(nn.Module):
    def __init__(self, n_classes, fusion_dim=FUSION_DIM, dropout=DROPOUT):
        super().__init__()
        self.gpt2     = GPT2Model.from_pretrained("gpt2")
        h = self.gpt2.config.hidden_size
        self.proj     = nn.Sequential(nn.Linear(h, fusion_dim),
                                      nn.LayerNorm(fusion_dim), nn.GELU())
        self.classifier = _make_head(fusion_dim, n_classes, dropout)

    def forward(self, gpt_ids, gpt_mask, **_):
        output = self.gpt2(input_ids=gpt_ids, attention_mask=gpt_mask)
        hs = last_real_token(output.last_hidden_state, gpt_mask)
        return self.classifier(self.proj(hs))

class AblationResNetOnly(nn.Module):
    def __init__(self, n_classes, fusion_dim=FUSION_DIM, dropout=DROPOUT):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.vision_features = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.vision_features.parameters():
            p.requires_grad = False
        self.vis_pool   = VisualAttentionPool(512)
        self.proj       = nn.Sequential(nn.Linear(512, fusion_dim),
                                        nn.LayerNorm(fusion_dim), nn.GELU())
        self.classifier = _make_head(fusion_dim, n_classes, dropout)

    def forward(self, images, image_mask, **_):
        B, N, C, H, W = images.shape
        with torch.no_grad():
            feat = self.vision_features(images.reshape(B * N, C, H, W)).reshape(B, N, -1)
        h = self.proj(self.vis_pool(feat, image_mask))
        return self.classifier(h)

LR           = 5e-5
WEIGHT_DECAY = 0.01

def make_optimizer(model):
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=WEIGHT_DECAY)

if not pdf_dir:
    raise FileNotFoundError(
        "Could not locate extracted_pdfs/data under /kaggle/input."
    )
texts, paths = load_corpus(pdf_dir)
if not texts:
    raise ValueError("The NUS PDF corpus is empty.")

labels, cluster_model = labels_sbert_kmeans(
    texts, n_clusters=N_CLUSTERS, seed=42
)
n_classes    = int(np.max(labels)) + 1

pd.DataFrame({"pdf_path": paths, "cluster_label": labels}).to_csv(
    os.path.join(OUT_DIR, "study3_cluster_labels.csv"), index=False
)

bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
gpt_tok = AutoTokenizer.from_pretrained("gpt2")
gpt_tok.pad_token = gpt_tok.eos_token
gpt_tok.padding_side = "left"
image_index = extract_images_from_pdfs(paths, cache_dir=IMG_CACHE,
                                       max_images_per_doc=N_IMAGES)

texts_arr  = np.asarray(texts, dtype=object)
labels_arr = np.asarray(labels)

def build_mm_loader(idx, shuffle, seed=None):
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    ds = MultimodalDataset(
        texts_arr[idx].tolist(), labels_arr[idx].tolist(),
        [paths[i] for i in idx], image_index,
        bert_tok, gpt_tok, max_len=MAX_LEN, n_images=N_IMAGES)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      generator=generator, pin_memory=True)

def eval_multimodal_model(model, loader):
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch  = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(batch["bert_ids"], batch["bert_mask"],
                           batch["gpt_ids"],  batch["gpt_mask"],
                           batch["images"],   batch["image_mask"])
            probs  = torch.softmax(logits, 1).cpu().numpy()
            y_probs.append(probs)
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            y_true.extend(batch["label"].cpu().numpy().tolist())
    return (np.array(y_true), np.array(y_pred), np.vstack(y_probs))

def eval_ablation_model(model, loader, model_type):
    model.eval()
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            if model_type == "bert":
                logits = model(bert_ids=batch["bert_ids"],
                               bert_mask=batch["bert_mask"])
            elif model_type == "gpt2":
                logits = model(gpt_ids=batch["gpt_ids"],
                               gpt_mask=batch["gpt_mask"])
            elif model_type == "resnet":
                logits = model(images=batch["images"],
                               image_mask=batch["image_mask"])
            else:
                logits = model(batch["bert_ids"], batch["bert_mask"],
                               batch["gpt_ids"],  batch["gpt_mask"],
                               batch["images"],   batch["image_mask"])
            probs = torch.softmax(logits, 1).cpu().numpy()
            y_probs.append(probs)
            y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
            y_true.extend(batch["label"].cpu().numpy().tolist())
    return (np.array(y_true), np.array(y_pred), np.vstack(y_probs))

def train_nn_model(model, loader, optimizer, loss_fn, epochs,
                   val_loader=None):
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for batch in loader:
            batch  = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            logits = model(batch["bert_ids"], batch["bert_mask"],
                           batch["gpt_ids"],  batch["gpt_mask"],
                           batch["images"],   batch["image_mask"])
            loss = loss_fn(logits, batch["label"])
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        train_loss = ep_loss / len(loader)
        if val_loader is not None:
            model.eval()
            vl = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch  = {k: v.to(DEVICE) for k, v in batch.items()}
                    logits = model(batch["bert_ids"], batch["bert_mask"],
                                   batch["gpt_ids"],  batch["gpt_mask"],
                                   batch["images"],   batch["image_mask"])
                    vl += loss_fn(logits, batch["label"]).item()
            print(f"    epoch {ep + 1}: train_loss={train_loss:.4f}  "
                  f"val_loss={vl / len(val_loader):.4f}")

def train_ablation_nn(model, loader, model_type, optimizer,
                      loss_fn, epochs):
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            optimizer.zero_grad()
            if model_type == "bert":
                logits = model(bert_ids=batch["bert_ids"],
                               bert_mask=batch["bert_mask"])
            elif model_type == "gpt2":
                logits = model(gpt_ids=batch["gpt_ids"],
                               gpt_mask=batch["gpt_mask"])
            elif model_type == "resnet":
                logits = model(images=batch["images"],
                               image_mask=batch["image_mask"])
            loss = loss_fn(logits, batch["label"])
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()

mm_name    = "Multimodal (BERT+GPT-2+ResNet-18)"
mm_pairs   = []
mm_per_seed = {}
loss_fn_mm = nn.CrossEntropyLoss()

for seed in SEEDS:
    set_seed(seed)
    idx = np.arange(len(texts))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2,
                                      stratify=labels_arr, random_state=seed)
    dl_tr = build_mm_loader(idx_tr, shuffle=True, seed=seed)
    dl_te = build_mm_loader(idx_te, shuffle=False)

    model = MultimodalBertGptResNet(n_classes=n_classes).to(DEVICE)
    opt   = make_optimizer(model)
    train_nn_model(model, dl_tr, opt, loss_fn_mm, MM_EPOCHS)

    y_t, y_p, y_s = eval_multimodal_model(model, dl_te)
    acc  = accuracy_score(y_t, y_p)
    _, _, f1, _ = precision_recall_fscore_support(y_t, y_p, average="weighted",
                                                   zero_division=0)
    mm_pairs.append((y_t, y_p, y_s))
    mm_per_seed[seed] = {
        "acc":   acc, "f1": f1,
        "auc":   compute_auc(y_t, y_s, n_classes),
        "brier": compute_brier(y_t, y_s, n_classes),
        "ece":   compute_ece(y_t, y_s),
    }
    del model; torch.cuda.empty_cache()

mm_summary, _ = metrics_from_pairs(mm_pairs, n_classes)

ablation_configs = [
    ("BERT only",          AblationBertOnly,    "bert"),
    ("GPT-2 only",         AblationGpt2Only,    "gpt2"),
    ("ResNet-18 only",     AblationResNetOnly,  "resnet"),
]
ablation_results  = {}
ablation_seed_accuracy = {}

for abl_name, ModelClass, mtype in ablation_configs:
    pairs = []
    seed_accuracy = {}
    for seed in SEEDS:
        set_seed(seed)
        idx = np.arange(len(texts))
        idx_tr, idx_te = train_test_split(idx, test_size=0.2,
                                          stratify=labels_arr, random_state=seed)
        dl_tr = build_mm_loader(idx_tr, shuffle=True, seed=seed)
        dl_te = build_mm_loader(idx_te, shuffle=False)

        model = ModelClass(n_classes=n_classes).to(DEVICE)
        opt   = make_optimizer(model)
        train_ablation_nn(model, dl_tr, mtype, opt, loss_fn_mm, MM_EPOCHS)
        
        y_t, y_p, y_s = eval_ablation_model(model, dl_te, mtype)
        pairs.append((y_t, y_p, y_s))
        seed_accuracy[seed] = accuracy_score(y_t, y_p)
        del model; torch.cuda.empty_cache()
        
    summary, _ = metrics_from_pairs(pairs, n_classes)
    ablation_results[abl_name]  = summary
    ablation_seed_accuracy[abl_name] = seed_accuracy

ablation_results[mm_name] = mm_summary

skf           = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results  = []
cv_true, cv_pred = [], []

for fold, (tr_idx, va_idx) in enumerate(skf.split(texts_arr, labels_arr)):
    set_seed(42 + fold)
    dl_tr = build_mm_loader(tr_idx, shuffle=True, seed=42 + fold)
    dl_va = build_mm_loader(va_idx, shuffle=False)
    
    model  = MultimodalBertGptResNet(n_classes=n_classes).to(DEVICE)
    opt    = make_optimizer(model)

    train_nn_model(model, dl_tr, opt, loss_fn_mm,
                   MM_EPOCHS, val_loader=dl_va)
    y_t, y_p, y_s = eval_multimodal_model(model, dl_va)
    
    acc  = accuracy_score(y_t, y_p)
    _, _, f1, _ = precision_recall_fscore_support(y_t, y_p, average="weighted",
                                                   zero_division=0)
    auc_v   = compute_auc(y_t, y_s, n_classes)
    brier_v = compute_brier(y_t, y_s, n_classes)
    ece_v   = compute_ece(y_t, y_s)
    fold_results.append({"fold": fold+1, "acc": acc, "f1": f1,
                          "auc": auc_v, "brier": brier_v, "ece": ece_v})
    cv_true.extend(y_t.tolist())
    cv_pred.extend(y_p.tolist())
    del model; torch.cuda.empty_cache()

per_seed_df = pd.DataFrame.from_dict(mm_per_seed, orient="index")
per_seed_df.index.name = "seed"
per_seed_df.reset_index().to_csv(
    os.path.join(OUT_DIR, "study3_multimodal_per_seed.csv"), index=False
)

summary_df = pd.DataFrame.from_dict(ablation_results, orient="index")
summary_df.index.name = "model"
summary_df.reset_index().to_csv(
    os.path.join(OUT_DIR, "study3_ablation_summary.csv"), index=False
)

proposed_accuracy = np.array([mm_per_seed[s]["acc"] for s in SEEDS])
wilcoxon_rows = []
for baseline_name, seed_values in ablation_seed_accuracy.items():
    baseline_accuracy = np.array([seed_values[s] for s in SEEDS])
    differences = proposed_accuracy - baseline_accuracy
    if np.allclose(differences, 0):
        statistic, p_value = np.nan, 1.0
        note = "all paired differences are zero"
    else:
        statistic, p_value = wilcoxon(
            proposed_accuracy,
            baseline_accuracy,
            alternative="greater",
            zero_method="wilcox",
        )
        note = ""
    wilcoxon_rows.append({
        "baseline": baseline_name,
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
        "significant_at_0.05": bool(p_value < 0.05),
        "note": note,
    })

pd.DataFrame(wilcoxon_rows).to_csv(
    os.path.join(OUT_DIR, "study3_wilcoxon.csv"), index=False
)

cv_df = pd.DataFrame(fold_results)
cv_df.to_csv(os.path.join(OUT_DIR, "study3_5fold_cv.csv"), index=False)

cv_cm = confusion_matrix(cv_true, cv_pred, labels=list(range(n_classes)))
pd.DataFrame(
    cv_cm,
    index=[f"true_{i}" for i in range(n_classes)],
    columns=[f"pred_{i}" for i in range(n_classes)],
).to_csv(os.path.join(OUT_DIR, "study3_cv_confusion_matrix.csv"))

print("\nProposed multimodal model — seven-seed summary")
print(pd.Series(mm_summary).to_string())
print("\nAblation summary")
print(summary_df.to_string())
print("\nWilcoxon tests")
print(pd.DataFrame(wilcoxon_rows).to_string(index=False))
print("\nFive-fold cross-validation")
print(cv_df.to_string(index=False))
print(
    f"Mean CV accuracy: {cv_df['acc'].mean():.4f} "
    f"+/- {cv_df['acc'].std(ddof=1):.4f}"
)
print("\nAggregated CV confusion matrix")
print(cv_cm)
print(f"\nAll outputs saved to {OUT_DIR}")
