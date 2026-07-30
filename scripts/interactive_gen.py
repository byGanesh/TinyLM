import torch
from tokenizers import Tokenizer
from src.model import TinyLM
from src.config import MAX_SEQ_LEN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TinyLM().to(device)
ckpt = torch.load("checkpoints/tinylm-v3-600M.pt", map_location=device)
model.load_state_dict(ckpt)
model.eval()
print(f"model loaded on {device}")

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

def generate(prompt, max_new_tokens=200, temperature=0.8, top_k=40):
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long).to(device)
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(out[0].tolist())

print("\nTinyLM Interactive Generation")
print("=" * 40)
print("commands:")
print("  /temp N    set temperature (default 0.8)")
print("  /topk N    set top-k (default 40)")
print("  /tokens N  set max new tokens (default 200)")
print("  /quit      exit")
print()

temperature = 0.8
top_k = 40
max_new = 200

while True:
    try:
        prompt = input(">>> ")
    except (EOFError, KeyboardInterrupt):
        break

    if not prompt.strip():
        continue
    if prompt.startswith("/quit"):
        break
    if prompt.startswith("/temp"):
        temperature = float(prompt.split()[1])
        print(f"  temperature = {temperature}")
        continue
    if prompt.startswith("/topk"):
        top_k = int(prompt.split()[1])
        print(f"  top_k = {top_k}")
        continue
    if prompt.startswith("/tokens"):
        max_new = int(prompt.split()[1])
        print(f"  max_new_tokens = {max_new}")
        continue

    print(generate(prompt, max_new_tokens=max_new,
                   temperature=temperature, top_k=top_k))
    print()
