"""
Phase 2: Computational Efficiency & GPU Latency Benchmark
Profiles total F-EDL model parameters, checkpoint storage size, batch latency,
amortized per-sample latency (at batch size 32), and throughput using CUDA events.
"""

import argparse
import os
import torch
from models.fedl_wrapper import FEDLWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="F-EDL Backbone Efficiency Profiler")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_warmup", type=int, default=20)
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    return parser.parse_args()


def benchmark_backbone(bb_key, display_name, ckpt_file, args, device):
    ckpt_path = os.path.join(args.checkpoint_dir, ckpt_file)
    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024) if os.path.exists(ckpt_path) else float("nan")

    # Instantiate complete F-EDL Wrapper model
    model = FEDLWrapper(backbone_name=bb_key, num_classes=7, pretrained=False).to(device)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)

    model.eval()

    # Total model parameter count (backbone + F-EDL heads)
    params_m = sum(p.numel() for p in model.parameters()) / 1e6

    dummy_input = torch.randn(args.batch_size, 3, 224, 224, device=device)

    # GPU Latency Benchmark using CUDA Events
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for _ in range(args.num_warmup):
            _ = model(dummy_input)

    torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(args.num_runs):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))

    avg_batch_latency_ms = float(sum(times) / len(times))
    amortized_sample_ms = avg_batch_latency_ms / args.batch_size
    throughput = (args.batch_size * 1000.0) / avg_batch_latency_ms

    return display_name, params_m, ckpt_size_mb, avg_batch_latency_ms, amortized_sample_ms, throughput


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    backbones = [
        ("efficientnet_b0", "EfficientNet-B0", f"ham10000_efficientnet_b0_fedl_standard_seed{args.seed}_best.pt"),
        ("shufflenet_v2", "ShuffleNetV2", f"ham10000_shufflenet_v2_fedl_standard_seed{args.seed}_best.pt"),
        ("mobilenet_v3_small", "MobileNetV3-Small", f"ham10000_mobilenet_v3_small_fedl_standard_seed{args.seed}_best.pt"),
    ]

    print("=" * 105)
    print(f"COMPUTATIONAL EFFICIENCY BENCHMARK (Hardware: {gpu_name})")
    print(f"Batch Size = {args.batch_size} | Warmup = {args.num_warmup} | Runs = {args.num_runs}")
    print("=" * 105)
    print(f"{'Backbone':<20}{'Params (M)':>12}{'Ckpt (MB)':>12}{'Batch Lat (ms)':>16}{'Amortized (ms/img)':>20}{'Throughput (img/s)':>20}")
    print("-" * 105)

    for bb_key, display_name, ckpt_file in backbones:
        res = benchmark_backbone(bb_key, display_name, ckpt_file, args, device)
        _, params, ckpt_size, batch_lat, amort_lat, thpt = res
        print(f"{display_name:<20}{params:>12.2f}{ckpt_size:>12.2f}{batch_lat:>16.2f}{amort_lat:>20.4f}{thpt:>20.1f}")

    print("=" * 105)
    print("Note: Latency measurements are hardware-dependent and reported for the specific GPU execution environment used.")


if __name__ == "__main__":
    main()