#!/usr/bin/env python3
"""
Export Gemma 4 E4B to ExecuTorch .pte format for Android (XNNPACK backend).

Quantization note:
  ExecuTorch uses XNNPACK 8da4w quantization (8-bit dynamic activation, 4-bit
  weight) — NOT the llama.cpp Q4_K_M GGUF format. 8da4w achieves a similar size
  footprint (~2.5 GB for E4B) while being directly runnable by the ExecuTorch
  runtime on Android/iOS.

Requirements:
  - GPU machine (A100 80GB or H100 recommended; minimum 24GB VRAM)
  - Python 3.10+
  - pip install torch==2.5.1 executorch torchtune transformers huggingface_hub

Usage:
  export HF_TOKEN=<your_token>  # Gemma 4 is gated; accept license at HF first
  python scripts/export_gemma4_e4b.py
  python scripts/export_gemma4_e4b.py --model-id google/gemma-4-4b-it --output-dir ./artifacts

After export, upload to bedda-tech HF org:
  huggingface-cli login
  huggingface-cli upload bedda-tech/react-native-executorch-gemma-4 ./artifacts/gemma-4-e4b/
"""

import argparse
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Gemma 4 E4B to ExecuTorch .pte")
    parser.add_argument(
        "--model-id",
        default="google/gemma-4-4b-it",
        help="HuggingFace model ID for Gemma 4 E4B instruct (default: google/gemma-4-4b-it)",
    )
    parser.add_argument(
        "--output-dir",
        default="./artifacts/gemma-4-e4b",
        help="Directory for exported artifacts",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        default=False,
        help="Skip 8da4w quantization and export full bfloat16 (requires ~8 GB device RAM)",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HuggingFace access token for gated model (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length for export (default: 2048)",
    )
    return parser.parse_args()


def check_prerequisites() -> None:
    """Verify required packages are available."""
    missing = []
    for pkg in ["torch", "executorch", "torchtune", "transformers", "huggingface_hub"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(
            f"ERROR: Missing packages: {', '.join(missing)}\n"
            "Run: pip install torch==2.5.1 executorch torchtune transformers huggingface_hub\n"
            "Or:  bash scripts/setup_export_env.sh"
        )
        sys.exit(1)

    import torch

    if not torch.cuda.is_available():
        print(
            "WARNING: No CUDA GPU detected.\n"
            "  Export will run on CPU — this takes many hours for a 4B model.\n"
            "  Recommended: A100 80GB, H100, or RTX 4090 (24GB VRAM minimum)."
        )
    else:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM)")
        if vram_gb < 24:
            print(
                f"WARNING: {vram_gb:.1f} GB VRAM detected. E4B bfloat16 needs ~16 GB minimum.\n"
                "  Export with --quantize (default) should fit; full bfloat16 may OOM."
            )


def export_model(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoTokenizer

    output_dir = Path(args.output_dir)
    quantize = not args.no_quantize
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Save tokenizer
    print(f"[1/5] Loading tokenizer from {args.model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        token=args.hf_token,
    )
    tokenizer.save_pretrained(output_dir)
    print(f"      Saved tokenizer to {output_dir}/")

    # Step 2: Load model via ExecuTorch LLM export utilities
    print(f"[2/5] Loading model weights from {args.model_id} ...")
    try:
        from executorch.extension.llm.export.builder import LLMEdgeManager
        from executorch.extension.llm.export.quantizer_lib import (
            get_pt2e_quantization_params,
            get_pt2e_quantizers,
        )
    except ImportError as e:
        print(
            f"ERROR: {e}\n"
            "executorch package is installed but LLM export utilities are missing.\n"
            "Build ExecuTorch from source with LLM support:\n"
            "  https://github.com/pytorch/executorch/blob/main/docs/source/llm/getting-started.md"
        )
        sys.exit(1)

    # Gemma 4 uses the gemma2 architecture from torchtune
    try:
        manager = LLMEdgeManager.from_model(
            checkpoint=args.model_id,
            model_class_name="gemma2",  # Gemma 4 shares the Gemma 2 architecture
            hf_token=args.hf_token,
            dtype=torch.bfloat16,
            use_kv_cache=True,
            enable_dynamic_shape=True,
            max_seq_len=args.seq_len,
        )
    except Exception as e:
        print(
            f"ERROR loading model: {e}\n"
            "Possible causes:\n"
            "  - Model ID is wrong (check HF: google/gemma-4-4b-it)\n"
            "  - HF token missing or Gemma 4 license not accepted\n"
            "  - Insufficient GPU memory\n"
            "  - torchtune doesn't yet have a gemma4 config (use gemma2 as fallback)"
        )
        sys.exit(1)

    # Step 3: Quantize (8da4w = 8-bit dynamic activation, 4-bit weight for XNNPACK)
    if quantize:
        print("[3/5] Applying XNNPACK 8da4w quantization ...")
        print("      (8-bit dynamic activation + 4-bit weight — ~2.5 GB output)")
        quant_params = get_pt2e_quantization_params(
            pt2e_quantize="xnnpack_dynamic",
        )
        quantizers = get_pt2e_quantizers(quant_params, quantize)
        manager = manager.capture_pre_autograd_graph().pt2e_quantize(quantizers)
    else:
        print("[3/5] Skipping quantization (bfloat16 export) ...")

    # Step 4: Export to .pte
    print("[4/5] Exporting to ExecuTorch .pte ...")
    if quantize:
        subdir = output_dir / "quantized"
        pte_filename = "gemma4_e4b_8da4w.pte"
    else:
        subdir = output_dir / "original"
        pte_filename = "gemma4_e4b_bf16.pte"

    subdir.mkdir(exist_ok=True)
    pte_path = subdir / pte_filename

    manager.export().to_executorch().save(str(pte_path))
    size_gb = pte_path.stat().st_size / 1e9
    print(f"      Saved: {pte_path} ({size_gb:.2f} GB)")

    # Step 5: Write manifest
    print("[5/5] Writing artifact manifest ...")
    manifest = {
        "model_id": args.model_id,
        "quantization": "8da4w" if quantize else "bf16",
        "backend": "xnnpack",
        "framework": "executorch",
        "seq_len": args.seq_len,
        "model_file": f"{'quantized' if quantize else 'original'}/{pte_filename}",
        "tokenizer_files": [
            "tokenizer.json",
            "tokenizer_config.json",
        ],
        "size_gb": round(size_gb, 2),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nDone. Artifacts in: {output_dir}/")
    _print_next_steps(output_dir)


def _print_next_steps(output_dir: Path) -> None:
    print("\nNext steps:")
    print("  1. Verify the .pte loads on Android:")
    print("     adb push artifacts/ /sdcard/deft-models/")
    print("     # Then test from the Deft app with GEMMA4_E4B_QUANTIZED constant")
    print()
    print("  2. Upload to bedda-tech HF org:")
    print("     huggingface-cli login")
    print(f"     huggingface-cli upload bedda-tech/react-native-executorch-gemma-4 {output_dir}/")
    print()
    print("  3. The react-native-executorch GEMMA4_E4B_QUANTIZED constant already")
    print("     points to the correct HF path. No code changes needed after upload.")


def main() -> None:
    args = parse_args()

    if not args.hf_token:
        print(
            "WARNING: HF_TOKEN not set.\n"
            "  Gemma 4 is a gated model — accept the license at:\n"
            "  https://huggingface.co/google/gemma-4-4b-it\n"
            "  Then set: export HF_TOKEN=hf_...\n"
        )

    check_prerequisites()
    export_model(args)


if __name__ == "__main__":
    main()
