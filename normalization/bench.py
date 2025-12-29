import argparse
import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from triton.testing import do_bench
import plotly.graph_objects as go

from rms_norm import rms_norm_fwd_tvm_ffi, rms_norm_kernel, ASSUMED_ALIGN_BYTES
from layer_norm import layer_norm_fwd
from quack.rmsnorm import rmsnorm_fwd as quack_rmsnorm_fwd


def get_system_info() -> dict:
    info = {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_arch": f"sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}",
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "num_gpus": torch.cuda.device_count(),
    }

    # Get peak bandwidth based on GPU
    gpu_name = info["gpu_name"].lower()
    if "b200" in gpu_name or "b100" in gpu_name:
        info["peak_bw"] = 8000  # GB/s
    elif "h100" in gpu_name:
        info["peak_bw"] = 3350  # GB/s (HBM3)
    elif "h200" in gpu_name:
        info["peak_bw"] = 4800  # GB/s
    elif "a100" in gpu_name:
        info["peak_bw"] = 2039  # GB/s (80GB HBM2e)
    else:
        info["peak_bw"] = None

    return info


def format_system_info(info: dict) -> str:
    lines = [
        f"GPU: {info['gpu_name']} ({info['gpu_arch']})",
        f"CUDA: {info['cuda_version']} | PyTorch: {info['torch_version']}",
    ]
    if info["peak_bw"]:
        lines.append(f"Peak Memory BW: {info['peak_bw']} GB/s")
    return "\n".join(lines)


def bench_rms_norm(M, N=128, dtype=torch.bfloat16):
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    w = torch.randn(N, device="cuda", dtype=dtype)
    eps = 1e-6

    with torch.inference_mode():
        ref = torch.nn.functional.rms_norm(x, (N,), weight=w, eps=eps)
        torch_time = do_bench(
            lambda: torch.nn.functional.rms_norm(x, (N,), weight=w, eps=eps)
        )

        compiled_torch = torch.compile(
            torch.nn.functional.rms_norm, dynamic=True, mode="reduce-overhead"
        )
        compiled_torch(x, (N,), weight=w, eps=eps)
        compiled_torch_time = do_bench(
            lambda: compiled_torch(x, (N,), weight=w, eps=eps)
        )

    with torch.inference_mode():
        quack_out, *_ = quack_rmsnorm_fwd(x, w, eps=eps)
        torch.testing.assert_close(quack_out, ref)
        quack_time = do_bench(lambda: quack_rmsnorm_fwd(x, w, eps=eps))

    y = rms_norm_fwd_tvm_ffi(x, w, eps)
    torch.testing.assert_close(y, ref)
    cute_wrapper_time = do_bench(lambda: rms_norm_fwd_tvm_ffi(x, w, eps))

    y_direct = torch.empty_like(x)
    x_cute = from_dlpack(x, assumed_align=ASSUMED_ALIGN_BYTES)
    y_cute = from_dlpack(y_direct, assumed_align=ASSUMED_ALIGN_BYTES)
    w_cute = from_dlpack(w, assumed_align=ASSUMED_ALIGN_BYTES)
    compiled_cute = cute.compile(rms_norm_kernel, x_cute, y_cute, w_cute, eps)
    compiled_cute(x_cute, y_cute, w_cute)
    torch.testing.assert_close(y_direct, ref)
    cute_direct_time = do_bench(lambda: compiled_cute(x_cute, y_cute, w_cute))

    bwd = lambda time: M * N * dtype.itemsize * 2 / time / 1e6

    return {
        "M": M,
        "torch": bwd(torch_time),
        "torch.compile": bwd(compiled_torch_time),
        "quack": bwd(quack_time),
        "cute": bwd(cute_wrapper_time),
        "cute (direct)": bwd(cute_direct_time),
    }


def bench_layer_norm(M, N=4096, dtype=torch.bfloat16):
    x = torch.randn(M, N, device="cuda", dtype=dtype)
    eps = 1e-6

    with torch.inference_mode():
        ref = torch.nn.functional.layer_norm(x, (N,), eps=eps)
        torch_time = do_bench(
            lambda: torch.nn.functional.layer_norm(x, (N,), eps=eps)
        )

        torch._dynamo.reset()
        compiled_torch = torch.compile(
            torch.nn.functional.layer_norm, mode="default", fullgraph=True
        )
        compiled_torch(x, (N,), eps=eps)
        torch.cuda.synchronize()
        compiled_torch_time = do_bench(
            lambda: compiled_torch(x, (N,), eps=eps)
        )

    y = layer_norm_fwd(x, eps)
    torch.testing.assert_close(y, ref)
    cute_time = do_bench(lambda: layer_norm_fwd(x, eps))

    bwd = lambda time: M * N * dtype.itemsize * 2 / time / 1e6

    return {
        "M": M,
        "torch": bwd(torch_time),
        "torch.compile": bwd(compiled_torch_time),
        "cute": bwd(cute_time),
    }


def run_benchmark(kernel: str, num_steps: int = 14):
    print(f"Benchmarking {kernel.upper()}")
    n_default = 128 if kernel == "rms" else 4096
    print(f"N={n_default}, dtype=bfloat16")
    print("-" * 90)

    bench_fn = bench_rms_norm if kernel == "rms" else bench_layer_norm
    results = []
    M = 1024
    for _ in range(num_steps):
        result = bench_fn(M)
        results.append(result)

        parts = [f"M={M:7d}"]
        for k, v in result.items():
            if k != "M":
                parts.append(f"{k}: {v:5.1f} GB/s")
        print(" | ".join(parts))

        M *= 2

    return results


def bench_layer_norm_heatmap(Ms: list[int], Ns: list[int], dtype=torch.bfloat16):
    results = []

    for N in Ns:
        row = []
        for M in Ms:
            x = torch.randn(M, N, device="cuda", dtype=dtype)
            eps = 1e-6

            with torch.inference_mode():
                torch._dynamo.reset()
                compiled_torch = torch.compile(
                    torch.nn.functional.layer_norm, mode="default", fullgraph=True
                )
                compiled_torch(x, (N,), eps=eps)
                torch.cuda.synchronize()
                torch_time = do_bench(lambda: compiled_torch(x, (N,), eps=eps))

            y = layer_norm_fwd(x, eps)
            cute_time = do_bench(lambda: layer_norm_fwd(x, eps))

            speedup = torch_time / cute_time
            bw_torch = M * N * dtype.itemsize * 2 / torch_time / 1e6
            bw_cute = M * N * dtype.itemsize * 2 / cute_time / 1e6

            row.append({
                "speedup": speedup,
                "torch_bw": bw_torch,
                "cute_bw": bw_cute,
            })
            print(f"M={M:6d} N={N:5d} | torch: {bw_torch:5.0f} GB/s | cute: {bw_cute:5.0f} GB/s | {speedup:.2f}x")
        results.append(row)

    return results


def plot_heatmap(results: list[list[dict]], Ms: list[int], Ns: list[int], output_path: str, metric: str = "speedup", sys_info: dict = None):
    import numpy as np

    # Extract the metric into a 2D array
    data = np.array([[r[metric] for r in row] for row in results])

    if metric == "speedup":
        title = "LayerNorm Speedup: CuTe vs torch.compile (bf16)"
        colorbar_title = "Speedup (x)"
    else:
        title = f"LayerNorm {metric} (bf16)"
        colorbar_title = "GB/s"

    # Add system info to title
    if sys_info:
        subtitle = f"<br><sup>{sys_info['gpu_name']} ({sys_info['gpu_arch']}) | CUDA {sys_info['cuda_version']} | PyTorch {sys_info['torch_version']}</sup>"
        title = title + subtitle

    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=[str(m) for m in Ms],
        y=[str(n) for n in Ns],
        text=[[f"{v:.2f}x" if metric == "speedup" else f"{v:.0f}" for v in row] for row in data],
        texttemplate="%{text}",
        textfont={"size": 10},
        colorscale="RdYlGn" if metric == "speedup" else "Viridis",
        colorbar=dict(title=colorbar_title),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Batch Size (M)",
        yaxis_title="Hidden Dimension (N)",
        width=900,
        height=600,
    )

    fig.write_image(output_path)
    print(f"\nHeatmap saved to {output_path}")


def plot_results(results: list[dict], kernel: str, output_path: str, n_dim: int, sys_info: dict = None):
    fig = go.Figure()

    Ms = [r["M"] for r in results]
    keys = [k for k in results[0].keys() if k != "M"]

    colors = {
        "torch": "#1f77b4",
        "torch.compile": "#ff7f0e",
        "quack": "#2ca02c",
        "cute": "#d62728",
        "cute (direct)": "#9467bd",
    }

    for key in keys:
        values = [r[key] for r in results]
        fig.add_trace(go.Scatter(
            x=Ms,
            y=values,
            mode="lines+markers",
            name=key,
            line=dict(color=colors.get(key, None), width=2),
            marker=dict(size=8),
        ))

    if sys_info and sys_info.get("peak_bw"):
        peak_bw = sys_info["peak_bw"]
        gpu_name = sys_info["gpu_name"].split()[1] if " " in sys_info["gpu_name"] else sys_info["gpu_name"]
        fig.add_trace(go.Scatter(
            x=Ms,
            y=[peak_bw] * len(Ms),
            mode="lines",
            name=f"{gpu_name} Peak ({peak_bw} GB/s)",
            line=dict(color="black", width=2, dash="dash"),
        ))

    title = f"{kernel.upper()} Norm - Memory Bandwidth vs Input Shape (N={n_dim}, bf16)"
    if sys_info:
        subtitle = f"<br><sup>{sys_info['gpu_name']} ({sys_info['gpu_arch']}) | CUDA {sys_info['cuda_version']} | PyTorch {sys_info['torch_version']}</sup>"
        title = title + subtitle

    fig.update_layout(
        title=title,
        xaxis_title="Input Shape M",
        yaxis_title="Memory Bandwidth (GB/s)",
        xaxis_type="log",
        yaxis_type="log",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        template="plotly_white",
        width=900,
        height=600,
    )

    fig.write_image(output_path)
    print(f"\nPlot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark normalization kernels")
    parser.add_argument(
        "kernel",
        choices=["rms", "layer", "all", "layer-heatmap"],
        help="Which kernel to benchmark",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=14,
        help="Number of benchmark steps (M doubles each step, starting at 1024)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plotly PNG visualization",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    sys_info = get_system_info()
    print(format_system_info(sys_info))
    print("=" * 70)

    if args.kernel == "layer-heatmap":
        Ms = [1024, 2048, 4096, 8192, 16384, 32768, 65536]
        Ns = [1024, 2048, 4096, 8192, 16384]
        print("LayerNorm Heatmap Benchmark")
        print("-" * 70)
        results = bench_layer_norm_heatmap(Ms, Ns)
        plot_heatmap(results, Ms, Ns, "layer_norm_heatmap.png", metric="speedup", sys_info=sys_info)
    elif args.kernel == "all":
        rms_results = run_benchmark("rms", args.steps)
        if args.plot:
            plot_results(rms_results, "rms", "rms_norm_bench.png", n_dim=128, sys_info=sys_info)
        print()
        layer_results = run_benchmark("layer", args.steps)
        if args.plot:
            plot_results(layer_results, "layer", "layer_norm_bench.png", n_dim=4096, sys_info=sys_info)
    else:
        n_dim = 128 if args.kernel == "rms" else 4096
        results = run_benchmark(args.kernel, args.steps)
        if args.plot:
            plot_results(results, args.kernel, f"{args.kernel}_norm_bench.png", n_dim=n_dim, sys_info=sys_info)
