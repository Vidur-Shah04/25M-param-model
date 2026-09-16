"""
Upgrade existing Phase-3 SFT checkpoints to a resume-capable format.

This does NOT retrain or modify model weights.
It reads the existing SFT checkpoint and the corresponding Phase-2
checkpoint, then adds reproducibility/resume metadata.

Run from the repository root:
    python upgrade_phase3_checkpoints.py

The original checkpoint is preserved as *.original.pt.
"""

import shutil
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent

LANGS = {
    "Gujarati": {
        "sft_candidates": [
            ROOT / "Gujarati" / "checkpoints" / "phase3_gujarati_sft.pt",
            ROOT / "Gujarati" / "checkpoints" / "phase3_gujarati_sft_final.pt",
            ROOT / "Gujarati" / "checkpoints" / "phase3_gujarati_sft_best.pt",
        ],
        "pretrain": ROOT / "Gujarati" / "checkpoints" / "checkpoint_030518_gujarati.pt",
        "tokenizer": ROOT / "Gujarati" / "tokenizer_clean" / "gujarati_10k.model",
        "sft_config": {
            "epochs": 5,
            "batch_size": 4,
            "grad_accumulation": 8,
            "learning_rate": 2e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
            "grad_clip": 1.0,
            "amp_enabled": True,
            "scheduler": None,
        },
    },
    "Assamese": {
        "sft_candidates": [
            ROOT / "Assamese" / "checkpoints" / "phase3_assamese_sft.pt",
            ROOT / "Assamese" / "checkpoints" / "phase3_assamese_sft_final.pt",
            ROOT / "Assamese" / "checkpoints" / "phase3_assamese_sft_best.pt",
        ],
        "pretrain": ROOT / "Assamese" / "checkpoints" / "checkpoint_030518_assamese.pt",
        "tokenizer": ROOT / "Assamese" / "tokenizer_clean" / "assamese_10k.model",
        "sft_config": {
            "epochs": 3,
            "batch_size": 4,
            "grad_accumulation": 8,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
            "grad_clip": 1.0,
            "amp_enabled": True,
            "scheduler": None,
        },
    },
}

def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

for language, info in LANGS.items():
    sft_path = first_existing(info["sft_candidates"])
    pretrain_path = info["pretrain"]

    if sft_path is None:
        print(f"[SKIP] {language}: no SFT checkpoint found.")
        continue

    if not pretrain_path.exists():
        raise FileNotFoundError(f"{language}: missing {pretrain_path}")

    sft = torch.load(sft_path, map_location="cpu", weights_only=False)
    pre = torch.load(pretrain_path, map_location="cpu", weights_only=False)

    if "model_state_dict" not in sft:
        raise KeyError(f"{language}: SFT checkpoint has no model_state_dict")
    if "optimizer_state_dict" not in sft:
        raise KeyError(f"{language}: SFT checkpoint has no optimizer_state_dict")
    if "config" not in pre:
        raise KeyError(f"{language}: Phase-2 checkpoint has no config")

    backup = sft_path.with_suffix(".original.pt")
    if not backup.exists():
        shutil.copy2(sft_path, backup)

    # Preserve every original field and add explicit resume metadata.
    upgraded = dict(sft)

    upgraded.update({
        "step": int(sft.get("step", sft.get("epoch", 0))),
        "scheduler_state_dict": sft.get("scheduler_state_dict", None),
        "scaler_state_dict": sft.get("scaler_state_dict", None),

        # Exact architecture used by the pretrained checkpoint.
        "config": pre["config"],

        # Explicit Phase-3 reproducibility metadata.
        "sft_config": info["sft_config"],
        "pretrained_checkpoint": str(pretrain_path),
        "tokenizer_path": str(info["tokenizer"]),
        "checkpoint_format_version": 2,

        "phase": 3,
        "stage": "SFT",
        "resume_note": (
            "Phase-3 SFT uses a fresh AdamW optimizer and no LR scheduler. "
            "scheduler_state_dict is therefore None. scaler_state_dict is "
            "None because the original SFT checkpoint did not persist GradScaler state."
        ),
    })

    # Write a new file first; never destroy the original.
    upgraded_path = sft_path.with_name(sft_path.stem + "_resume.pt")
    torch.save(upgraded, upgraded_path)

    # Verify immediately.
    check = torch.load(upgraded_path, map_location="cpu", weights_only=False)
    required = [
        "step",
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "scaler_state_dict",
        "config",
        "sft_config",
        "phase",
        "stage",
    ]
    missing = [k for k in required if k not in check]
    if missing:
        raise RuntimeError(f"{language}: verification failed; missing {missing}")

    print(f"\n[{language}] PASS")
    print("Original :", sft_path)
    print("Backup   :", backup)
    print("Upgraded :", upgraded_path)
    print("Epoch    :", check["epoch"])
    print("Params   :", sum(v.numel() for v in check["model_state_dict"].values()
                           if hasattr(v, "numel")))
    print("Config   :", check["config"])
    print("Scheduler:", check["scheduler_state_dict"])
    print("Scaler   :", check["scaler_state_dict"])

print("\nDone. Review the *_resume.pt files before replacing/renaming the final checkpoints.")
