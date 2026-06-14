"""GPU smoke for the pure-Mamba VARC config: build the model, run a forward+backward
at the real patch=2 / image-size 64 resolution (32x32=1024 tokens), check shapes.
Validates config instantiation, mamba_ssm import, and the Mamba2 CUDA kernel."""
import torch
from src.ARC_HyenaResNet import build_hyena_arc_resnet, HyenaResNetVARCWrapper

CFG = "/home/dwessel/code/nvSubquadratic-private/varc_configs/cfg_mamba_p2_bidir.py"
dev = "cuda"
print("[smoke] building model from", CFG)
net = build_hyena_arc_resnet(CFG, num_tasks=400)
model = HyenaResNetVARCWrapper(net).to(dev)
nparams = sum(p.numel() for p in model.parameters())
print(f"[smoke] params = {nparams/1e6:.2f}M")

B, H, W = 4, 64, 64
x = torch.randint(0, 12, (B, H, W), device=dev)
task_ids = torch.randint(0, 400, (B,), device=dev)
print("[smoke] forward...")
logits = model(x, task_ids)
print("[smoke] logits", tuple(logits.shape), logits.dtype)
assert logits.shape[0] == B and logits.shape[1] == 12, logits.shape
print("[smoke] backward...")
loss = logits.float().mean()
loss.backward()
ng = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
print(f"[smoke] params with nonzero grad: {ng}")
print("[smoke] SMOKE_OK")
