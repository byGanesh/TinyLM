import os
import re
import torch
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from config import SPECIAL_TOKENS, VOCAB_SIZE

def clean_text(txt):
    txt = re.sub(r'https?://\S+|www\.\S+', '', txt)
    txt = re.sub(r'<.*?>', '', txt)
    txt = re.sub(r'\S+@\S+', '', txt)
    txt = re.sub(r'\s+', " ", txt)
    txt = txt.strip()
    return txt

def train_tokenizer(corpus_path):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )

    tokenizer.train([corpus_path], trainer)
    os.makedirs("tokenizer", exist_ok=True)
    tokenizer.save("tokenizer/tokenizer.json")
    print(f"tokenizer trained. vocab size is {tokenizer.get_vocab_size()}")
    return tokenizer

def tokenize_data(corpus_path, tokenizer):
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()

    docs = [d.strip() for d in text.split('\n\n') if len(d.strip()) > 0]

    all_ids = []
    for doc in docs:
        ids = tokenizer.encode(doc).ids
        if len(ids) < 10:
            continue
        all_ids.extend(ids)
    return np.array(all_ids, dtype=np.uint16)

def save_bin(arr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr.tofile(path)
    print(f"Saved {len(arr):} tokens {path}")

def build_dataset(
    raw_path = "data/corpus.txt",
    cleaned_path = "data/cleaned_corpus.txt",
    train_path = "data/train.bin",
    val_path = "data/val.bin",
    val_split = 0.05
):
    print("Cleaning...")
    with open(raw_path, "r", encoding="utf-8") as f:
        raw = f.read()

    cleaned = clean_text(raw)
    with open(cleaned_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print("training tokenizer...")
    tokenizer = train_tokenizer(cleaned_path)

    print("tokenizing...")
    tokens = tokenize_data(cleaned_path, tokenizer)
    print(f"Total tokens: {len(tokens):,}")

    split = int(len(tokens) * (1 - val_split))
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    print("Saving...")
    save_bin(train_tokens, train_path)
    save_bin(val_tokens, val_path)

    print(f"train: {len(train_tokens):,} tokens")
    print(f"val: {len(val_tokens):,} tokens")

def load_bin(path):
    data = np.memmap(path, dtype=np.uint16, mode='r')
    return torch.tensor(data, dtype=torch.long)

if __name__ == "__main__":
    build_dataset()
