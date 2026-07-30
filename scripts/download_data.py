from datasets import load_dataset
import os

os.makedirs("data", exist_ok=True)

def download_streaming(dataset_name, config, output_path, max_chars=500_000_000):
    print(f"streaming {dataset_name}/{config}...")

    ds = load_dataset(dataset_name, config, split="train", streaming=True)

    char_count = 0
    article_count = 0

    with open(output_path, "a", encoding="utf-8") as f:
        for article in ds:
            text = article["text"].strip()
            if len(text) == 0:
                continue
            f.write(text + "\n\n")
            char_count += len(text)
            article_count += 1

            if article_count % 10000 == 0:
                print(f"  {article_count} articles | {char_count/1e6:.0f}MB", end="\r")

            if char_count >= max_chars:
                print(f"\n  stopped at {max_chars/1e6:.0f}MB limit")
                break

    print(f"done — {article_count} articles | {char_count/1e6:.0f}MB")

# TinyStories
def download_tinystories(output_path):
    print("streaming TinyStories...")
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    char_count = 0
    count = 0

    with open(output_path, "a", encoding="utf-8") as f:
        for story in ds:
            text = story["text"].strip()
            if len(text) == 0:
                continue
            f.write(text + "\n\n")
            char_count += len(text)
            count += 1
            if count % 50000 == 0:
                print(f"  {count} stories | {char_count/1e6:.0f}MB", end="\r")

    print(f"done — {count} stories | {char_count/1e6:.0f}MB")


# Fineweb
def download_fineweb(output_path, max_chars=500_000_000):
    print("streaming FineWeb-Edu...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name      = "sample-10BT",
        split     = "train",
        streaming = True,
    )
    char_count = 0
    count = 0
    with open(output_path, "a", encoding="utf-8") as f:
        for doc in ds:
            text = doc["text"].strip()
            if len(text) == 0:
                continue
            f.write(text + "\n\n")
            char_count += len(text)
            count += 1
            if count % 10000 == 0:
                print(f"  {count} docs | {char_count/1e6:.0f}MB", end="\r")
            if char_count >= max_chars:
                print(f"\n  stopped at {max_chars/1e6:.0f}MB limit")
                break
    print(f"done — {count} docs | {char_count/1e6:.0f}MB")


download_streaming(
    "wikimedia/wikipedia",
    "20231101.simple",
    "data/corpus.txt",
    max_chars=200_000_000,   # 200MB
)

download_tinystories("data/corpus.txt")


download_streaming(
    "wikimedia/wikipedia",
    "20231101.en",
    "data/corpus.txt",
    max_chars=300_000_000,
)

download_fineweb("data/corpus.txt", max_chars=500_000_000)
