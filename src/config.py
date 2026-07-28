# Vocabulary
VOCAB_SIZE = 16000
PAD_TOKEN_ID = 0

# Model dimensions
D_MODEL = 256
N_LAYERS = 12
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS
FFN_DIM = 768

# Sequence length
MAX_SEQ_LEN = 1024

# RoPE
ROPE_THETA = 10000.0

# Regularization
DROPOUT = 0.0

# Weight tying
TIE_EMB = True

# Training
BATCH_SIZE = 16
MAX_LR = 3e-3
MIN_LR = 3e-4
WARMUP_STEPS = 2000
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

# Total params (approx)
# Embedding: VOCAB_SIZE x D_MODEL = 2.0M
# Attention: N_LAYERS x (4 x D_MODEL x D_MODEL) = 3.1M (Q,K,V, O Projections)
# FFN: N_LAYERS x (3 x D_MODEL x FFN_DIM) = 7.1M (gate, up, down)
# Norms: small (negligible)
# TOTAL: ~ 12-13M params
