import torch
import time
from src.model import TinyLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

model = TinyLM().to(device).to(dtype)
model.eval()

B, T = 1, 512
x = torch.randint(0, 100, (B, T), device=device)

with torch.autocast(device_type="cuda", dtype=dtype):
    # warmup
    for _ in range(10):
        model(x)

    # benchmark forward
    torch.cuda.synchronize()
    start = time.perf_counter()
    n_runs = 100
    for _ in range(n_runs):
        model(x)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

avg_ms = elapsed / n_runs * 1000
tokens_per_sec = B * T * n_runs / elapsed
params = model.num_params()

print(f"device:         {device}")
print(f"dtype:          {dtype}")
print(f"parameters:     {params:,}")
print(f"batch_size:     {B}")
print(f"seq_len:        {T}")
print(f"avg forward:    {avg_ms:.2f} ms")
print(f"tokens/sec:     {tokens_per_sec:,.0f}")
print(f"memory:         {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
