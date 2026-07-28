import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (FFN_DIM, HEAD_DIM, MAX_SEQ_LEN, N_HEADS, N_KV_HEADS, N_QUERIES_PER_KV, PAD_TOKEN_ID, ROPE_THETA, TIE_EMB, VOCAB_SIZE, D_MODEL, N_LAYERS)
import math

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.gamma

def precompute_freqs_cis(h_dim, max_seq, theta):
    freqs = 1.0 / (theta ** (torch.arange(0, h_dim, 2).float() / h_dim))
    pos = torch.arange(max_seq).float()
    angles = torch.outer(pos, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def rotary_emb(q,k,freqs_cis):
    q_c = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_c = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    freqs = freqs_cis.unsqueeze(0).unsqueeze(2)
    q_out = torch.view_as_real(q_c * freqs).flatten(3).type_as(q)
    k_out = torch.view_as_real(k_c * freqs).flatten(3).type_as(k)
    return q_out, k_out


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(D_MODEL, FFN_DIM, bias = False)
        self.up = nn.Linear(D_MODEL, FFN_DIM, bias = False)
        self.down = nn.Linear(FFN_DIM, D_MODEL, bias = False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

# Mult-Head Attention
class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(D_MODEL, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(D_MODEL, N_HEADS * HEAD_DIM, bias = False)
        self.v_proj = nn.Linear(D_MODEL, N_HEADS * HEAD_DIM, bias = False)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL, bias = False)

    def forward(self, x, freqs_cis):
        B, T, _ = x.shape
        Q = self.q_proj(x).view(B, T, N_HEADS, HEAD_DIM)
        K = self.k_proj(x).view(B, T, N_KV_HEADS, HEAD_DIM)
        V = self.v_proj(x).view(B,T, N_KV_HEADS, HEAD_DIM)

        Q, K = rotary_emb(Q, K, freqs_cis)

        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)

        out = F.scaled_dot_product_attention(Q,K,V, is_causal = True)
        out = out.transpose(1,2).contiguous().view(B,T, D_MODEL)
        return self.o_proj(out)

class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_attn = RMSNorm(D_MODEL)
        self.attn = Attention()
        self.norm_ff = RMSNorm(D_MODEL)
        self.ff = FeedForward()

    def forward(self, x, freqs_cis):
       x = x + self.attn(self.norm_attn(x), freqs_cis)
       x = x + self.ff(self.norm_ff(x))
       return x

class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])
        self.norm = RMSNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias = False)

        if TIE_EMB:
            self.lm_head.weight = self.tok_emb.weight

        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(HEAD_DIM, MAX_SEQ_LEN, ROPE_THETA)
        )

        self._init_weights()

    def _init_weights(self):
        for md in self.modules():
            if isinstance(md, nn.Linear):
                nn.init.normal_(md.weight, mean=0.0, std = 0.02)
            elif isinstance(md, nn.Embedding):
                nn.init.normal_(md.weight, mean = 0.0, std =0.02)

        scale = 1.0 / math.sqrt(2* N_LAYERS)
        for name, param in self.named_parameters():
            if 'o_proj.weight' in name or 'down.weight' in name:
                param.data.mul_(scale)

    def forward(self, idx, tg= None):
        B, T = idx.shape
        x = self.tok_emb(idx)
        freqs = self.freqs_cis[:T]

        for l in self.layers:
            x = l(x,freqs)

        x = self.norm(x)

        if tg is not None: # target
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE),
                tg.view(-1),
                ignore_index = PAD_TOKEN_ID,
            )
            return logits, loss

        # inference
        logits = self.lm_head(x[:, [-1], :])
        return logits, None

    # disabiling grad in generation
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, stop_token_id = None):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -MAX_SEQ_LEN:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, next_token], dim=1)
            if stop_token_id is not None and (next_token == stop_token_id).all():
                break
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
