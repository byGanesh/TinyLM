import os
import math
from src.config import BATCH_SIZE, GRAD_CLIP, MAX_LR, MAX_SEQ_LEN, MIN_LR, WARMUP_STEPS, WEIGHT_DECAY
import torch
from src.model import TinyLM
from torch.optim import AdamW

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

# lr scheduling
def get_lr(step, total_steps):
    if step < WARMUP_STEPS:
        return MAX_LR * (step / WARMUP_STEPS)

    progress = (step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return MIN_LR + (MAX_LR - MIN_LR) * cos

# data loading
def get_batch(data, step):
    ix = torch.randint(len(data) - MAX_SEQ_LEN, (BATCH_SIZE,))
    x = torch.stack([data[i: i + MAX_SEQ_LEN] for i in ix])
    y = torch.stack([data[i + 1: i + MAX_SEQ_LEN + 1] for i in ix])
    return x.to(device), y.to(device)


# Optimizer
def mk_optimizer(model):
    decay = [p for n, p in model.named_parameters() if p.dim() >=2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(groups, lr=MAX_LR, betas=(0.9, 0.95), eps=1e-8)

# eval
@torch.no_grad()
def evaluate(model, val_data, eval_steps=50):
    model.eval()
    losses = []
    for i in range(eval_steps):
        x, y = get_batch(val_data, i)
        with torch.autocast(device_type = "cuda", dtype = dtype):
            _, loss = model(x,y)
        losses.append(loss.item())

    model.train()
    return sum(losses) / len(losses)


# checkpoint
def save_checkpoint(model, optimizer, step, loss, path="checkpoints", keep_last=3):
    os.makedirs(path, exist_ok=True)
    m = model.module if hasattr(model, 'module') else model
    torch.save({
        "step":      step,
        "model":     m.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss":      loss,
    }, f"{path}/ckpt_{step}.pt")
    print(f"saved checkpoint at step {step}")

    ckpts = sorted([
        f for f in os.listdir(path) if f.startswith("ckpt_")
    ], key=lambda x: int(x.split("_")[1].split(".")[0]))

    for ck in ckpts[:-keep_last]:
        os.remove(f"{path}/{ck}")
        print(f"deleted old checkpoint: {ck}")


def load_checkpoint(model, optimizer, path):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print(f"resumed from step {ckpt['step']}, loss {ckpt['loss']:.4f}")
    return ckpt["step"]

# main training loop
def train(
    train_data,
    val_data,
    total_steps = 30000,
    eval = 500,
    save = 5000,
    resume_from = None,
    log = 50,
):
    model = TinyLM().to(device)

    if torch.cuda.device_count > 1:
        print(f"using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    optimizer = mk_optimizer(model)

    start_step = 0
    if resume_from:
        start_step = load_checkpoint(model, optimizer, resume_from)

    print(f"params: {model.num_params():,}")
    print(f"training on {device} | type {dtype}")
    print(f"total steps: {total_steps} | tokens: {total_steps * BATCH_SIZE * MAX_SEQ_LEN:,}")

    model.train()
    running_loss = 0.0

    for step in range(start_step, total_steps):
        # LR update
        lr = get_lr(step, total_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # Forward and backward
        x, y = get_batch(train_data, step)

        with torch.autocast(device_type = "cuda", dtype = dtype):
            _, loss = model(x,y)

        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        optimizer.zero_grad(set_to_none = True)
        running_loss += loss.item()

        # loggin
        if step % log == 0 and step > 0:
            avg_loss = running_loss / log
            print(f"step {step:6d} | loss {avg_loss:.4f} | ppl {math.exp(avg_loss):7.2f} | lr {lr:.2e}")
            running_loss = 0.0

        # evaluation
        if step % eval == 0 and step > 0:
            val_loss = evaluate(model, val_data)
            print(f"  >> val loss {val_loss:.4f} | val ppl {math.exp(val_loss):.2f}")

        # checkpoint
        if step % save == 0 and step > 0:
            save_checkpoint(model, optimizer, step, loss.item())

    save_checkpoint(model, optimizer, total_steps, loss.item())
    print("training complete")
    m = model.module if hasattr(model, 'module') else model
    torch.save(
        m.model.state_dict(),
        "checkpoints/tinylm_final.pt"
    )
    print("saved final model weights...")
    return model
