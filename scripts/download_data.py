from datasets import load_dataset
import os

os.makedirs("data", exist_ok=True)

print("downloading Simple Wikipedia...")
ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")

print("writing to data/corpus.txt...")
with open("data/corpus.txt", "w", encoding="utf-8") as f:
    for article in ds:
        text = article["text"].strip()
        if len(text) > 0:
            f.write(text + "\n\n")

print(f"done: {len(ds)} articles written")
