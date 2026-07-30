from datasets import load_dataset
import os

os.makedirs("data", exist_ok=True)

def download_streaming(dataset_name, config, output_path, max_tokens=100_000_000):
    print(f"streaming {dataset_name}...")

    ds = load_dataset(dataset_name, config, split="train", streaming=True)

    char_count = 0
    article_cout = 0

    with open(output_path, "a", encoding="utf-8") as f:
        for art in ds:
            txt = art["text"].strip()
            if len(txt) == 0:
                continue
            f.write(txt + "\n\n")
            char_count += len(txt)
            article_cout += 1

            if article_cout % 10000 == 0:
                print(f"{article_cout} articles, {char_count/1e6:.0f}MB", end="\r")

            if char_count > 500_000_000:
                break
    print(f"\n - {article_cout} articles, {char_count/1e6:.0f}MB")

download_streaming(
    "wikimedia/wikipedia",
    "20231101.en",
    "data/corpus.txt"
)
