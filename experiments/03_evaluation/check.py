import os, time, torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Visible GPUs: {torch.cuda.device_count()}  (CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')})")

if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    raise SystemExit("No GPUs visible to PyTorch.")

# Small GEMM on each GPU
sz = 1024  # keep it fast
for i in range(torch.cuda.device_count()):
    d = torch.device(f"cuda:{i}")
    props = torch.cuda.get_device_properties(d)
    total_mem_gb = props.total_memory / (1024**3)
    print(f"\n[GPU {i}] {props.name}  {total_mem_gb:.1f} GiB, CC {props.major}.{props.minor}")

    torch.cuda.synchronize(d)
    t0 = time.time()
    a = torch.randn(sz, sz, device=d, dtype=torch.float32)
    b = torch.randn(sz, sz, device=d, dtype=torch.float32)
    c = a @ b
    loss = c.norm()
    loss.backward()  # triggers another kernel
    torch.cuda.synchronize(d)
    dt = (time.time() - t0) * 1000
    print(f"[GPU {i}] matmul+backward ok, norm={loss.item():.4f}, elapsed={dt:.1f} ms")

print("\nAll GPUs ran a kernel successfully ✅")
