import torch
from tokenizers import Tokenizer
from src.model import TinyLM
from src.config import MAX_SEQ_LEN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load
model = TinyLM().to(device)
ckpt  = torch.load("checkpoints/tinylm-v1-66M.pt", map_location=device)
model.load_state_dict(ckpt)
model.eval()

tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

def generate(prompt, max_new_tokens=100, temperature=0.8, top_k=40):
    ids = tokenizer.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long).to(device)
    out = model.generate(idx, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())

print(generate("The most important thing in life is"))
print("---")
print(generate("Language is"))
print("---")
print(generate("I do not know"))
