from datasets import load_dataset
import os

os.makedirs("data", exist_ok=True)

print("downloading Simple wikipedia...")
ds = load_dataset("wikipedia", "20220301.simple", split="train", trust_remote_code=True)

print("writing to corpus.txt...")
with open("data/corpus.txt", "w", encoding="utf-8") as f:
    for article in ds:
        txt = article["text"].strip()
        if len(txt) > 0:
            f.write(txt + "\n\n")
print(f"done {len(ds)} articles written")
