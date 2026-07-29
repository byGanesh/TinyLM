import os
import re
import torch
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from src.config import MAX_SEQ_LEN, SPECIAL_TOKENS, VOCAB_SIZE

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
    raw_path   = "data/corpus.txt",
    train_path = "data/train.bin",
    val_path   = "data/val.bin",
    val_split  = 0.05,
):
        print("building tokenizer corpus...")
        tok_corpus = "data/tok_corpus.txt"
        with open(raw_path, "r", encoding="utf-8") as f_in, \
             open(tok_corpus, "w", encoding="utf-8") as f_out:
            chars = 0
            for line in f_in:
                f_out.write(line)
                chars += len(line)
                if chars > 50_000_000:
                    break
        print(f"tokenizer corpus ready: {chars/1e6:.1f}MB")

        print("training tokenizer...")
        tokenizer = train_tokenizer(tok_corpus)

        print("tokenizing...")
        temp_path = "data/all_tokens.bin"
        total = 0

        with open(raw_path, "r", encoding="utf-8") as f, \
             open(temp_path, "ab") as out:
            doc = []
            for line in f:
                line = line.strip()
                if line == "":
                    if doc:
                        text = clean_text(" ".join(doc))
                        ids  = tokenizer.encode(text).ids
                        if len(ids) >= 10:
                            np.array(ids, dtype=np.uint16).tofile(out)
                            total += len(ids)
                        doc = []
                    if total % 500_000 == 0 and total > 0:
                        print(f"  {total:,} tokens...", end="\r")
                else:
                    doc.append(line)

        print(f"\ntotal tokens: {total:,}")

        print("splitting and saving...")
        all_tokens = np.memmap(temp_path, dtype=np.uint16, mode='r')
        split = int(len(all_tokens) * (1 - val_split))

        save_bin(np.array(all_tokens[:split]), train_path)
        save_bin(np.array(all_tokens[split:]), val_path)

        os.remove(temp_path)
        print(f"\ndone.")
        print(f"train: {split:,} tokens")
        print(f"val:{total - split:,} tokens")
        print(f"teps at batch=16, seq={MAX_SEQ_LEN}: {split // (16 * MAX_SEQ_LEN):,}")


def load_bin(path):
    data = np.memmap(path, dtype=np.uint16, mode='r')
    return torch.tensor(np.array(data), dtype=torch.long)

if __name__ == "__main__":
    build_dataset()
