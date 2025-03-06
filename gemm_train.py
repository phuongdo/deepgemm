import torch
import torch.nn as nn
import torch.optim as optim
import time
import math

# Import functions from deep_gemm
from deep_gemm import calc_diff, ceil_div, get_col_major_tma_aligned_tensor, gemm_fp8_fp8_bf16_nt

def quantize_fp8_lhs(x, block_size=128):
    """
    Vectorized quantization of input tensor x (FP32) to a simulated FP8 representation.
    Quantizes along the last dimension in blocks.
    Returns a tuple (x_q, scales) where:
      - x_q has the same shape as x, containing quantized values (simulated FP8).
      - scales has shape [..., num_blocks].
    """
    orig_shape = x.shape
    in_features = orig_shape[-1]
    # Compute padding size to make the last dimension divisible by block_size
    pad_size = (block_size - (in_features % block_size)) % block_size
    if pad_size:
        x = F.pad(x, (0, pad_size), mode='constant', value=0)
    # Reshape to [..., num_blocks, block_size]
    new_shape = orig_shape[:-1] + ((in_features + pad_size) // block_size, block_size)
    x_reshaped = x.view(new_shape)
    # Compute scales along the block dimension
    scales = x_reshaped.abs().amax(dim=-1)
    scales[scales == 0] = 1.0
    # Quantize: scale, round, and clip to [-127, 127]
    x_q = torch.round(x_reshaped / scales.unsqueeze(-1) * 127)
    x_q = torch.clamp(x_q, -127, 127)
    # Reshape quantized tensor back to original shape (remove padding if added)
    x_q = x_q.view(-1, (in_features + pad_size))[:, :in_features].view(orig_shape)
    scales = scales.view(*orig_shape[:-1], -1)
    # Cast quantized tensor to simulated FP8 type
    x_q = x_q.to(torch.float8_e4m3fn)
    return x_q, scales

def quantize_fp8_rhs(x, block_size=128):
    """
    Vectorized quantization of weight tensor x (FP32) to a simulated FP8 representation.
    Quantizes in blocks along both dimensions.
    Returns a tuple (x_q, scales) where:
      - x_q has the same shape as x.
      - scales is of shape [ceil(out_features/block_size), ceil(in_features/block_size)].
    """
    out_features, in_features = x.shape
    # Compute padding sizes for rows and columns
    pad_rows = (block_size - (out_features % block_size)) % block_size
    pad_cols = (block_size - (in_features % block_size)) % block_size
    if pad_rows or pad_cols:
        x = F.pad(x, (0, pad_cols, 0, pad_rows), mode='constant', value=0)
    new_out = x.shape[0] // block_size
    new_in = x.shape[1] // block_size
    # Reshape and permute to group blocks together
    x_reshaped = x.view(new_out, block_size, new_in, block_size).permute(0, 2, 1, 3)
    # Compute scales for each block
    scales = x_reshaped.abs().amax(dim=(-1, -2))
    scales[scales == 0] = 1.0
    # Quantize each block
    x_q = torch.round(x_reshaped / scales.unsqueeze(-1).unsqueeze(-1) * 127)
    x_q = torch.clamp(x_q, -127, 127)
    # Permute and reshape back to original dimensions (remove padding if added)
    x_q = x_q.permute(0, 2, 1, 3).contiguous().view(x.shape)
    if pad_rows or pad_cols:
        x_q = x_q[:out_features, :in_features]
    x_q = x_q.to(torch.float8_e4m3fn)
    return x_q, scales

# Custom FP8 Linear layer using our deep_gemm kernel.
class FP8Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, block_size=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

    def forward(self, x):
        orig_shape = x.shape  # Save original shape
        x_flat = x.view(-1, orig_shape[-1])
        lhs_q, lhs_scales = quantize_fp8_lhs(x_flat, self.block_size)
        rhs_q, rhs_scales = quantize_fp8_rhs(self.weight, self.block_size)
        out_tensor = torch.empty(x_flat.shape[0], self.out_features, dtype=torch.bfloat16, device=x.device)
        gemm_fp8_fp8_bf16_nt((lhs_q, lhs_scales), (rhs_q, rhs_scales), out_tensor)
        out_fp32 = out_tensor.to(torch.float32)
        if self.bias is not None:
            out_fp32 = out_fp32 + self.bias
        new_shape = orig_shape[:-1] + (self.out_features,)
        return out_fp32.view(new_shape)

# Simple Transformer-like block using either FP32 or FP8 linear layers.
class SimpleTransformer(nn.Module):
    def __init__(self, d_model, d_ff, use_fp8=False):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)
        if use_fp8:
            self.linear1 = FP8Linear(d_model, d_ff)
            self.linear2 = FP8Linear(d_ff, d_model)
        else:
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x_norm = self.layernorm(x)
        ff = self.linear2(self.activation(self.linear1(x_norm)))
        return x + ff  # residual connection

# Complete Transformer model with an embedding layer, transformer block, and classifier head.
class SimpleTransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_classes, use_fp8=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = SimpleTransformer(d_model, d_ff, use_fp8=use_fp8)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [batch, seq_len]
        x = self.embedding(x)         # [batch, seq_len, d_model]
        x = self.transformer(x)         # [batch, seq_len, d_model]
        x = x.mean(dim=1)              # simple mean pooling
        logits = self.classifier(x)
        return logits

def train_model(model, optimizer, criterion, device, num_iters=50, batch_size=32, seq_len=16):
    model.train()
    total_loss = 0.0
    start = time.time()
    for _ in range(num_iters):
        # Create synthetic inputs and targets
        inputs = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        targets = torch.randint(0, 10, (batch_size,), device=device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    end = time.time()
    avg_loss = total_loss / num_iters
    elapsed = end - start
    return avg_loss, elapsed

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)

    # Hyperparameters
    vocab_size = 1000
    d_model = 128
    d_ff = 256
    num_classes = 10
    num_iters = 50
    batch_size = 32
    seq_len = 16

    # Create FP32 (standard) model
    model_fp32 = SimpleTransformerModel(vocab_size, d_model, d_ff, num_classes, use_fp8=False).to(device)
    optimizer_fp32 = optim.SGD(model_fp32.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Create FP8 (mixed precision) model
    model_fp8 = SimpleTransformerModel(vocab_size, d_model, d_ff, num_classes, use_fp8=True).to(device)
    optimizer_fp8 = optim.SGD(model_fp8.parameters(), lr=0.01)

    print("Training FP32 model...")
    loss_fp32, time_fp32 = train_model(model_fp32, optimizer_fp32, criterion, device,
                                       num_iters=num_iters, batch_size=batch_size, seq_len=seq_len)
    print(f"FP32 Model: Avg Loss = {loss_fp32:.4f}, Time for {num_iters} iters = {time_fp32:.4f} s, "
          f"{time_fp32/num_iters*1e3:.2f} ms per iter")

    print("\nTraining FP8 (mixed precision) model...")
    loss_fp8, time_fp8 = train_model(model_fp8, optimizer_fp8, criterion, device,
                                     num_iters=num_iters, batch_size=batch_size, seq_len=seq_len)
    print(f"FP8 Model:  Avg Loss = {loss_fp8:.4f}, Time for {num_iters} iters = {time_fp8:.4f} s, "
          f"{time_fp8/num_iters*1e3:.2f} ms per iter")

if __name__ == "__main__":
    main()
