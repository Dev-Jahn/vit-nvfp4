"""Throughput benchmark: NVFP4 W4A4 vs BF16 vs FP8 dense GEMM on sm_120.

Measures the pure kernel (operands pre-quantized/packed once), reporting TFLOP/s.
Run: TORCH_CUDA_ARCH_LIST=12.0a uv run python bench/bench_gemm.py
"""
import torch
from torch.nn.functional import ScalingType, SwizzleType

from vit_nvfp4.nvfp4.quant import quantize_to_nvfp4
from vit_nvfp4.nvfp4.pack import pack_e2m1, as_float4_x2, to_blocked

# (M, K, N) — ViT-representative GEMM shapes.
SHAPES = [
    (256, 1024, 1024), (1024, 1024, 1024),
    (256, 1152, 4304), (1024, 1152, 4304),
    (1024, 1408, 6144), (4096, 1408, 6144),
    (4096, 4096, 4096),
]
_BLK = [ScalingType.BlockWise1x16, ScalingType.TensorWise]
_SWZ = [SwizzleType.SWIZZLE_32_4_4, SwizzleType.NO_SWIZZLE]


def _time(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters / 1e3  # seconds


def _prep_fp4(t):  # t: (R, K) row-major -> packed fp4 (R, K//2) + swizzled scale + global
    codes, bs, gs = quantize_to_nvfp4(t, 16)
    return as_float4_x2(pack_e2m1(codes.contiguous())), to_blocked(bs.contiguous()), gs.reshape(1).float()


def bench():
    dev = "cuda"
    print(f"{'M':>5} {'K':>5} {'N':>5} | {'fp4':>8} {'bf16':>8} {'fp8':>8}  (TFLOP/s) | {'fp4/bf16':>9} {'fp4/fp8':>8}")
    print("-" * 80)
    for M, K, N in SHAPES:
        a = torch.randn(M, K, device=dev)
        b = torch.randn(N, K, device=dev)  # weight, row-major (N,K)
        flops = 2 * M * N * K

        # NVFP4 (pre-packed; time pure kernel)
        a_fp4, a_sf, a_g = _prep_fp4(a)
        b_fp4, b_sf, b_g = _prep_fp4(b)
        b_fp4 = b_fp4.t()  # (K//2, N)
        t_fp4 = _time(lambda: torch._scaled_mm_v2(
            a_fp4, b_fp4, [a_sf, a_g], _BLK, _SWZ, [b_sf, b_g], _BLK, _SWZ, None, torch.bfloat16))

        # BF16
        ab, bb = a.bfloat16(), b.bfloat16().t()
        t_bf16 = _time(lambda: ab @ bb)

        # FP8 e4m3 per-tensor
        try:
            a8 = a.to(torch.float8_e4m3fn)
            b8 = b.to(torch.float8_e4m3fn).t()
            one = torch.ones(1, device=dev, dtype=torch.float32)
            t_fp8 = _time(lambda: torch._scaled_mm(a8, b8, one, one, out_dtype=torch.bfloat16))
            fp8s = f"{flops / t_fp8 / 1e12:8.1f}"
            ratio8 = f"{t_fp8 / t_fp4:8.2f}"
        except Exception as e:
            fp8s, ratio8 = f"{'n/a':>8}", f"{'n/a':>8}"

        print(f"{M:>5} {K:>5} {N:>5} | {flops/t_fp4/1e12:8.1f} {flops/t_bf16/1e12:8.1f} {fp8s}  "
              f"| {t_bf16/t_fp4:9.2f} {ratio8}")


if __name__ == "__main__":
    print("torch", torch.__version__, "| device", torch.cuda.get_device_name(0))
    bench()
