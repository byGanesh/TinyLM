from src.data import build_dataset, load_bin
from src.train import train

build_dataset()

train_data = load_bin("data/train.bin")
val_data   = load_bin("data/val.bin")

print(train_data.shape)
print(val_data.shape)

train(train_data, val_data, total_steps=100)
