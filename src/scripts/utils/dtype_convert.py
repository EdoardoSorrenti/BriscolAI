"""Utility to convert a saved BriscolAI checkpoint between torch dtypes.

Usage example:
    python convert_to_fp32.py --input models/270k_0_3.pth --target-dtype float32

By default the script writes a sibling file whose name reflects the target dtype.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

_DTYPE_ALIASES = {
    "float32": torch.float32,
    "float": torch.float32,
    "fp32": torch.float32,
    "single": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float64": torch.float64,
    "double": torch.float64,
    "fp64": torch.float64,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
}

try:
    from config import save_path as DEFAULT_INPUT_PATH
except Exception:
    DEFAULT_INPUT_PATH = "models/270k_0_3.pth"


def parse_dtype(dtype_str: str) -> torch.dtype:
    key = dtype_str.strip().lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(
            f"Unsupported dtype '{dtype_str}'. Known options: {', '.join(sorted(_DTYPE_ALIASES))}"
        )
    return _DTYPE_ALIASES[key]


def _cast_to_dtype(obj: Any, dtype: torch.dtype) -> Any:
    """Recursively cast every tensor in ``obj`` to the target dtype."""
    if torch.is_tensor(obj):
        return obj.to(dtype)
    if isinstance(obj, dict):
        return {key: _cast_to_dtype(value, dtype) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_cast_to_dtype(value, dtype) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_cast_to_dtype(value, dtype) for value in obj)
    if isinstance(obj, set):
        return {_cast_to_dtype(value, dtype) for value in obj}
    return obj


def _resolve_output_path(input_path: Path, target_dtype: torch.dtype, user_supplied: str | None) -> Path:
    if user_supplied:
        return Path(user_supplied).expanduser().resolve()

    suffix = input_path.suffix or ".pth"
    stem = input_path.stem
    dtype_name = str(target_dtype).split(".")[-1]
    output_name = f"{stem}_{dtype_name}{suffix}"
    return input_path.with_name(output_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert every tensor inside a checkpoint to the requested dtype."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Path to the source checkpoint (default: config.save_path).",
    )
    parser.add_argument("--output", type=str, default=None, help="Destination for the converted checkpoint.")
    parser.add_argument(
        "--target-dtype",
        type=str,
        default="float32",
        help="Torch dtype to cast tensors to (e.g. float32, bfloat16, float16, float64).",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Optionally load the converted state dict into PolicyNetwork to verify compatibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    target_dtype = parse_dtype(args.target_dtype)
    output_path = _resolve_output_path(input_path, target_dtype, args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input checkpoint not found: {input_path}")

    checkpoint = torch.load(input_path, map_location="cpu")
    converted = _cast_to_dtype(checkpoint, target_dtype)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output_path)

    if args.check:
        from model import PolicyNetwork

        state_dict: dict[str, torch.Tensor]
        if isinstance(converted, dict) and "state_dict" in converted:
            state_dict = converted["state_dict"]
        elif isinstance(converted, dict):
            state_dict = {k: v for k, v in converted.items() if torch.is_tensor(v)}
            if not state_dict:
                raise ValueError(
                    "Converted checkpoint doesn't contain tensors when --check is used; "
                    "please specify a compatible format."
                )
        else:
            raise ValueError("--check expects a dict checkpoint containing a state_dict.")

        model = PolicyNetwork().to(dtype=torch.float32)
        # Cast state_dict to fp32 for loading purposes if target dtype is not float32
        loadable_state_dict = {k: v.to(torch.float32) if torch.is_tensor(v) else v for k, v in state_dict.items()}
        model.load_state_dict(loadable_state_dict, strict=False)

    print(f"Converted checkpoint written to {output_path}")


if __name__ == "__main__":
    main()
