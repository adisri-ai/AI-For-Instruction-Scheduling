"""
train_model.py

... (docstring remains the same) ...
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ADDED: Import warnings and numpy for summary
import warnings
import re
import ast
import argparse
import pickle
import json
from typing import List, Tuple, Dict, Any, Optional
import numpy as np  # ADDED
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from training import build_cfg
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.layers import Input, GlobalAveragePooling1D, Attention
from tensorflow.keras.models import Model

# ADDED: Suppress common Keras warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _safe_eval_condition(cond_text: str, ctx: Dict[str, int]) -> Optional[int]:
    # ... (function is unchanged)
    try:
        t = cond_text.strip()
        if t.startswith("if"):
            t = t[2:].strip()
        if t.startswith("(") and t.endswith(")"):
            t = t[1:-1].strip()
        for var, val in ctx.items():
            t = re.sub(rf"\b{re.escape(var)}\b", str(int(val)), t)
        node = ast.parse(t, mode="eval")
        allowed = (
            ast.Expression,
            ast.Compare,
            ast.BinOp,
            ast.Num,
            ast.UnaryOp,
            ast.NameConstant,
            ast.Load,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.Mod,
            ast.Pow,
            ast.Lt,
            ast.Gt,
            ast.Eq,
            ast.LtE,
            ast.GtE,
            ast.NotEq,
            ast.BitAnd,
            ast.BitOr,
            ast.And,
            ast.Or,
            ast.BoolOp,
            ast.Constant,
        )
        for n in ast.walk(node):
            if not isinstance(n, allowed):
                return None
        val = eval(compile(node, "<string>", "eval"))
        return 1 if bool(val) else 0
    except Exception:
        return None


def _parse_for_iterations(header: str, ctx: Dict[str, int]) -> Optional[int]:
    # ... (function is unchanged)
    try:
        inner = header[header.find("(") + 1 : header.rfind(")")]
        parts = [p.strip() for p in inner.split(";")]
        if len(parts) >= 2:
            cond = parts[1]
            for var, val in ctx.items():
                cond = re.sub(rf"\b{re.escape(var)}\b", str(int(val)), cond)
            m = re.search(r"<\s*(-?\d+)", cond)
            if m:
                return int(m.group(1))
        return None
    except Exception:
        return None


def _extract_assignments_from_nodes(nodes: List[str]) -> Dict[str, int]:
    # ... (function is unchanged)
    ctx: Dict[str, int] = {}
    assignment_re = re.compile(r"^\s*(\w+)\s*=\s*(-?\d+)\s*;?\s*$")

    for node_text in nodes:
        lines = node_text.splitlines()
        for ln in lines:
            ln = ln.strip()
            match = assignment_re.match(ln)
            if match:
                var = match.group(1)
                val_str = match.group(2)
                try:
                    ctx[var] = int(val_str)
                except ValueError:
                    pass
    return ctx


def extract_training_df_from_dataset(
    csv_path: str, max_context_nodes: int = 20
) -> pd.DataFrame:
    # ... (function is unchanged)
    df = pd.read_csv(csv_path)
    if "code_str" not in df.columns:
        raise ValueError("data.csv must contain 'code_str' column")

    records = []
    for idx, row in df.iterrows():
        code = str(row["code_str"])
        node_dict, edges = build_cfg(code)
        nodes = [node_dict[i] for i in range(len(node_dict))]

        visited = []
        last_if_chain_results = []

        i = 0
        steps = 0
        while i < len(nodes) and steps < max(200, len(nodes) * 4):
            steps += 1
            node = nodes[i]
            if node.startswith("if(") or node.startswith("else if("):
                ctx = _extract_assignments_from_nodes(visited)
                inp_nodes = visited[-max_context_nodes:]
                inp = " || ".join(inp_nodes) if inp_nodes else "<EMPTY_CONTEXT>"
                val = _safe_eval_condition(node, ctx)
                if val is None:
                    val = 0
                records.append({"input": inp, "label": int(val)})
                last_if_chain_results.append(int(val))
                visited.append(node)
                i += 1
                continue
            if node.strip().startswith("else") or node.startswith("else"):
                ctx = _extract_assignments_from_nodes(visited)
                inp_nodes = visited[-max_context_nodes:]
                inp = " || ".join(inp_nodes) if inp_nodes else "<EMPTY_CONTEXT>"
                val = (
                    1
                    if (
                        len(last_if_chain_results) > 0
                        and sum(last_if_chain_results) == 0
                    )
                    else 0
                )
                records.append({"input": inp, "label": int(val)})
                visited.append(node)
                last_if_chain_results = []
                i += 1
                continue
            if node.startswith("for("):
                ctx = _extract_assignments_from_nodes(visited)
                inp_nodes = visited[-max_context_nodes:]
                inp = " || ".join(inp_nodes) if inp_nodes else "<EMPTY_CONTEXT>"
                iters = _parse_for_iterations(node, ctx)
                val = 1 if (iters is not None and iters > 0) else 0
                records.append({"input": inp, "label": int(val)})
                visited.append(node)
                i += 1
                continue
            visited.append(node)
            i += 1
    out = pd.DataFrame.from_records(records)
    if out.empty:
        print("Warning: no training records were extracted from dataset.")
    return out


def tokenize_and_pad(
    texts: List[str],
    tokenizer: Optional[Tokenizer] = None,
    maxlen: Optional[int] = None,
) -> Tuple[np.ndarray, Tokenizer, int]:
    # ... (function is unchanged)
    if tokenizer is None:
        tokenizer = Tokenizer(oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
    seqs = tokenizer.texts_to_sequences(texts)
    if maxlen is None:
        maxlen = max(1, max(len(s) for s in seqs))
    padded = pad_sequences(seqs, maxlen=maxlen, padding="pre")
    return padded, tokenizer, maxlen


def build_and_train_model(
    X: np.ndarray,
    y: np.ndarray,
    vocab_size: int,
    embed_dim: int = 64,
    lstm_units: int = 64,
    epochs: int = 8,
    batch_size: int = 32,
    save_prefix: str = "cfg_lstm",
) -> Dict[str, Any]:
    """
    Builds the Attention-based LSTM model, trains it, and saves artifacts.
    """
    maxlen = X.shape[1]

    inputs = Input(shape=(maxlen,))
    embedding_layer = Embedding(
        input_dim=vocab_size + 1, output_dim=embed_dim, input_length=maxlen
    )(inputs)
    lstm_out = LSTM(lstm_units, return_sequences=True)(embedding_layer)
    attention_out = Attention()([lstm_out, lstm_out])
    pooling_out = GlobalAveragePooling1D()(attention_out)
    dropout_out = Dropout(0.3)(pooling_out)
    outputs = Dense(1, activation="sigmoid")(dropout_out)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    # CHANGED: verbose=2 back to verbose=1 to show progress bar
    history = model.fit(
        X,
        y,
        validation_split=0.1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1,
    )

    model_path = f"{save_prefix}_model.keras"
    tokenizer_path = f"{save_prefix}_tokenizer.pkl"
    meta_path = f"{save_prefix}_meta.pkl"

    model.save(model_path)

    meta = {
        "maxlen": int(maxlen),
        "vocab_size": int(vocab_size),
        "embed_dim": int(embed_dim),
        "lstm_units": int(lstm_units),
        "model_path": model_path,
        "meta_path": meta_path,
        "tokenizer_path": tokenizer_path,
    }

    return {"model": model, "history": history.history, "meta": meta}


def save_tokenizer(tokenizer: Tokenizer, path: str):
    # ... (function is unchanged)
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)


def save_meta(meta: Dict[str, Any], path: str):
    # ... (function is unchanged)
    with open(path, "wb") as f:
        pickle.dump(meta, f)


def train_from_csv(
    csv_path: str,
    save_prefix: str = "cfg_lstm",
    epochs: int = 8,
    embed_dim: int = 64,
    lstm_units: int = 64,
):
    """
    High-level training function.
    """
    print(f"Reading dataset from {csv_path} ...")
    df = extract_training_df_from_dataset(csv_path)
    if df.empty:
        raise RuntimeError("No training records were extracted — check data.csv format")

    texts = df["input"].astype(str).tolist()
    labels = df["label"].astype(int).values

    X, tokenizer, maxlen = None, None, None
    X, tokenizer, maxlen = tokenize_and_pad(texts, tokenizer=None, maxlen=None)
    vocab_size = len(tokenizer.word_index)

    print(f"Training model for up to {epochs} epochs...")
    result = build_and_train_model(
        X,
        labels,
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        lstm_units=lstm_units,
        epochs=epochs,
        save_prefix=save_prefix,
    )

    model = result["model"]
    meta = result["meta"]
    history = result["history"]

    # Save tokenizer and meta
    tok_path = f"{save_prefix}_tokenizer.pkl"
    meta_path = f"{save_prefix}_meta.pkl"
    save_tokenizer(tokenizer, tok_path)

    meta.update({"tokenizer_path": tok_path, "maxlen": int(maxlen)})
    save_meta(meta, meta_path)

    # ... (rest of the summary print is unchanged)
    print("\n--- Training Complete ---")

    best_epoch_idx = np.argmin(history["val_loss"])
    best_val_loss = history["val_loss"][best_epoch_idx]
    best_val_acc = history["val_accuracy"][best_epoch_idx]
    best_train_acc = history["accuracy"][best_epoch_idx]
    epoch_num = best_epoch_idx + 1
    total_epochs = len(history["val_loss"])

    print(f"Early stopping triggered after {total_epochs} epochs.")
    print(f"Restoring best weights from Epoch {epoch_num}.")
    print("-------------------------")
    print(f"  Best Validation Loss: {best_val_loss:.4f}")
    print(f"  Best Validation Acc:  {best_val_acc:.4f}")
    print(f"  Best Training Acc:    {best_train_acc:.4f}")
    print("-------------------------")
    print(f"Saved: model -> {meta['model_path']}")
    print(f"Saved: tokenizer -> {tok_path}")
    print(f"Saved: meta -> {meta_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # --- FIX was here: changed 'add.argument' to 'add_argument' ---
    parser.add_argument("--csv", type=str, default="data.csv")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--lstm_units", type=int, default=64)
    parser.add_argument("--save_prefix", type=str, default="cfg_lstm")
    args, unknown = parser.parse_known_args()

    train_from_csv(
        args.csv,
        save_prefix=args.save_prefix,
        epochs=args.epochs,
        embed_dim=args.embed_dim,
        lstm_units=args.lstm_units,
    )
