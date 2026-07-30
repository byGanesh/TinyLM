import torch
from tokenizers import Tokenizer
from src.model import TinyLM
from src.config import MAX_SEQ_LEN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TinyLM().to(device)

checkpoint = torch.load("checkpoints/ckpt_24000.pt", map_location=device)

model.load_state_dict(checkpoint["model"])
model.eval()

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")


def generate(prompt, max_new_tokens=150, temperature=0.8, top_k=50):
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

    return tokenizer.decode(out[0].tolist())


prompts = [
    "Once upon a time",
    "The most important thing in life is",
    "I do not know",
    "Language is",
    "A person who always learns",
]

for p in prompts:
    print(f">>> {p}")
    print(generate(p))
    print("---")
