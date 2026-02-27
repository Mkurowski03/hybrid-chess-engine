"""Extract raw model weights from a training checkpoint."""
import torch
import sys

ckpt = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
out = sys.argv[2]
torch.save(ckpt["model"], out)
print(f"Extracted model weights from epoch {ckpt['epoch']} (best_loss={ckpt['best_loss']:.4f}) -> {out}")
