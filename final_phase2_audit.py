import sys
import os
import math
import importlib.util
from pathlib import Path

import torch


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

MODELS = {
    "GUJARATI": {
        "dir": PROJECT_DIR / "Gujarati",
        "checkpoint": PROJECT_DIR / "Gujarati" / "checkpoints" / "checkpoint_030518_gujarati.pt",
        "tokenizer": PROJECT_DIR / "Gujarati" / "tokenizer_clean" / "gujarati_24k.model",
        "model_dir": PROJECT_DIR / "Gujarati" / "model",
    },
    "ASSAMESE": {
        "dir": PROJECT_DIR / "Assamese",
        "checkpoint": PROJECT_DIR / "Assamese" / "checkpoints" / "checkpoint_030518_assamese.pt",
        "tokenizer": PROJECT_DIR / "Assamese" / "tokenizer_clean" / "assamese_24k.model",
        "model_dir": PROJECT_DIR / "Assamese" / "model",
    },
}


# ============================================================
# EXPECTED CONFIGURATION
# ============================================================

EXPECTED_CONFIG = {
    "vocab_size": 24000,
    "d_model": 512,
    "n_layers": 5,
    "n_heads": 8,
    "d_ff": 1536,
    "context_length": 512,
    "dropout": 0.1,
}

EXPECTED_STEP = 30518
EXPECTED_PARAMS = 25_689_088


# ============================================================
# HELPERS
# ============================================================

total_tests = 0
passed_tests = 0
failed_tests = 0


def check(condition, name, details=""):
    global total_tests, passed_tests, failed_tests

    total_tests += 1

    if condition:
        passed_tests += 1
        print(f"  [PASS] {name}" + (f" - {details}" if details else ""))
    else:
        failed_tests += 1
        print(f"  [FAIL] {name}" + (f" - {details}" if details else ""))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def load_module(module_name, file_path):
    """
    Load a Python file under a unique module name.

    This prevents Gujarati's `model` package from being reused
    accidentally when Assamese is audited afterward.
    """
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(file_path),
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create module spec for {file_path}")

    module = importlib.util.module_from_spec(spec)

    # Register under unique name before executing.
    sys.modules[module_name] = module

    spec.loader.exec_module(module)

    return module


def load_model_modules(model_dir, prefix):
    """
    Load config.py and transformer.py under a temporary package
    so that transformer.py can correctly resolve:

        from .config import ModelConfig

    Each language gets its own unique package name, preventing
    Gujarati and Assamese modules from being mixed.
    """

    config_path = model_dir / "config.py"
    transformer_path = model_dir / "transformer.py"

    if not config_path.exists():
        raise FileNotFoundError(config_path)

    if not transformer_path.exists():
        raise FileNotFoundError(transformer_path)

    package_name = f"{prefix}_model_package"

    # Create a unique package module.
    import types

    package = types.ModuleType(package_name)
    package.__path__ = [str(model_dir)]
    package.__package__ = package_name

    sys.modules[package_name] = package

    # Load config.py as package_name.config
    config_spec = importlib.util.spec_from_file_location(
        f"{package_name}.config",
        str(config_path),
        submodule_search_locations=None,
    )

    if config_spec is None or config_spec.loader is None:
        raise ImportError(f"Could not load {config_path}")

    config_module = importlib.util.module_from_spec(config_spec)
    config_module.__package__ = package_name

    sys.modules[f"{package_name}.config"] = config_module

    config_spec.loader.exec_module(config_module)

    # Load transformer.py as package_name.transformer
    transformer_spec = importlib.util.spec_from_file_location(
        f"{package_name}.transformer",
        str(transformer_path),
        submodule_search_locations=None,
    )

    if transformer_spec is None or transformer_spec.loader is None:
        raise ImportError(f"Could not load {transformer_path}")

    transformer_module = importlib.util.module_from_spec(transformer_spec)
    transformer_module.__package__ = package_name

    sys.modules[f"{package_name}.transformer"] = transformer_module

    transformer_spec.loader.exec_module(transformer_module)

    return config_module, transformer_module
def make_test_input(vocab_size, batch_size=1, sequence_length=32):
    """
    Deterministic input for reproducible sanity testing.
    """
    return torch.arange(
        batch_size * sequence_length,
        dtype=torch.long,
        device=device,
    ).reshape(batch_size, sequence_length) % vocab_size


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("FINAL PHASE 2 MODEL AUDIT")
print("=" * 75)

print(f"Project directory: {PROJECT_DIR}")
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("GPU: CPU")


# ============================================================
# AUDIT EACH MODEL
# ============================================================

for language, paths in MODELS.items():

    print("\n")
    print("=" * 75)
    print(f"{language} MODEL AUDIT")
    print("=" * 75)

    model_dir = paths["model_dir"]
    checkpoint_path = paths["checkpoint"]
    tokenizer_path = paths["tokenizer"]

    # --------------------------------------------------------
    # FILE CHECKS
    # --------------------------------------------------------

    print("\nFile checks:")

    check(
        paths["dir"].exists(),
        "Model directory exists",
        str(paths["dir"]),
    )

    check(
        checkpoint_path.exists(),
        "Checkpoint exists",
        str(checkpoint_path),
    )

    check(
        tokenizer_path.exists(),
        "Tokenizer exists",
        str(tokenizer_path),
    )

    check(
        model_dir.exists(),
        "Transformer directory exists",
        str(model_dir),
    )

    # --------------------------------------------------------
    # LOAD MODULES
    # --------------------------------------------------------

    print("\nModule loading:")

    try:
        prefix = language.lower()

        config_module, transformer_module = load_model_modules(
            model_dir,
            prefix,
        )

        check(
            True,
            "Model modules import",
            f"{config_module.__name__}, {transformer_module.__name__}",
        )

    except Exception as e:
        check(
            False,
            "Model modules import",
            repr(e),
        )
        continue

    # --------------------------------------------------------
    # CHECKPOINT
    # --------------------------------------------------------

    print("\nCheckpoint:")

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        check(
            True,
            "Checkpoint loads",
        )

    except Exception as e:
        check(
            False,
            "Checkpoint loads",
            repr(e),
        )
        continue

    required_keys = [
        "step",
        "model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "config",
        "loss",
        "val_loss",
        "learning_rate",
        "dataset_state",
    ]

    missing_keys = [
        key for key in required_keys
        if key not in checkpoint
    ]

    check(
        len(missing_keys) == 0,
        "Checkpoint contains required states",
        "all required keys present"
        if not missing_keys
        else f"missing={missing_keys}",
    )

    step = checkpoint.get("step", None)

    check(
        step == EXPECTED_STEP,
        "Checkpoint step",
        f"step={step}",
    )

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    print("\nConfiguration:")

    saved_config = checkpoint.get("config", None)

    if saved_config is None:
        check(
            False,
            "Configuration exists in checkpoint",
        )
        continue

    # Convert dataclass / namespace / dictionary safely.
    if hasattr(saved_config, "__dict__"):
        config_values = vars(saved_config)
    elif isinstance(saved_config, dict):
        config_values = saved_config
    else:
        config_values = {}

    for key, expected in EXPECTED_CONFIG.items():

        actual = config_values.get(key, None)

        check(
            actual == expected,
            key,
            f"actual={actual}, expected={expected}",
        )

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    print("\nTokenizer:")

    try:
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()

        loaded = sp.load(str(tokenizer_path))

        check(
            loaded,
            "Tokenizer loads",
        )

        vocab_size = sp.get_piece_size()

        check(
            vocab_size == EXPECTED_CONFIG["vocab_size"],
            "Tokenizer vocabulary = 24K",
            f"vocab_size={vocab_size}",
        )

    except Exception as e:
        check(
            False,
            "Tokenizer loads",
            repr(e),
        )
        sp = None

    # --------------------------------------------------------
    # MODEL CONSTRUCTION
    # --------------------------------------------------------

    print("\nModel construction:")

    try:
        ModelConfig = config_module.ModelConfig
        DecoderOnlyTransformer = transformer_module.DecoderOnlyTransformer

        config = ModelConfig(**config_values)

        model = DecoderOnlyTransformer(config)

        model = model.to(device)

        check(
            True,
            "Model construction",
        )

    except Exception as e:
        check(
            False,
            "Model construction",
            repr(e),
        )
        continue

    # --------------------------------------------------------
    # LOAD WEIGHTS
    # --------------------------------------------------------

    print("\nModel weights:")

    try:
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        check(
            True,
            "Model weights load",
        )

    except Exception as e:
        check(
            False,
            "Model weights load",
            repr(e),
        )
        continue

    # --------------------------------------------------------
    # PARAMETER COUNT
    # --------------------------------------------------------

    param_count = count_parameters(model)

    check(
        param_count == EXPECTED_PARAMS,
        "Parameter count",
        f"{param_count:,}",
    )

    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------

    print("\nForward-pass checks:")

    try:
        model.eval()

        x = make_test_input(
            vocab_size=EXPECTED_CONFIG["vocab_size"],
            batch_size=1,
            sequence_length=32,
        )

        y = torch.roll(x, shifts=-1, dims=1)

        with torch.no_grad():
            logits, loss = model(x, targets=y)

        expected_shape = (
            1,
            32,
            EXPECTED_CONFIG["vocab_size"],
        )

        check(
            tuple(logits.shape) == expected_shape,
            "Logits shape",
            f"actual={tuple(logits.shape)}, expected={expected_shape}",
        )

        check(
            torch.isfinite(logits).all().item(),
            "Logits finite",
        )

        check(
            torch.isfinite(loss).item(),
            "Loss finite",
            f"loss={loss.item():.6f}",
        )

    except Exception as e:
        check(
            False,
            "Forward pass",
            repr(e),
        )
        continue

    # --------------------------------------------------------
    # ATTENTION EXTRACTION
    # --------------------------------------------------------

    print("\nAttention extraction:")

    try:
        attention_result = model(
            x,
            targets=y,
            return_attention=True,
        )

        check(
            isinstance(attention_result, tuple)
            and len(attention_result) == 3,
            "Attention-enabled forward returns 3 values",
        )

        attention_logits, attention_loss, attention_maps = attention_result

        check(
            len(attention_maps) == EXPECTED_CONFIG["n_layers"],
            "Number of attention maps",
            f"layers={len(attention_maps)}",
        )

        attention_ok = True

        for layer_index, attn in enumerate(attention_maps):

            if not torch.isfinite(attn).all().item():
                attention_ok = False

            expected_attention_shape = (
                1,
                EXPECTED_CONFIG["n_heads"],
                32,
                32,
            )

            if tuple(attn.shape) != expected_attention_shape:
                attention_ok = False

        check(
            attention_ok,
            "Attention shapes and values",
            f"expected per layer=(1, {EXPECTED_CONFIG['n_heads']}, 32, 32)",
        )

    except Exception as e:
        check(
            False,
            "Attention extraction",
            repr(e),
        )

    # --------------------------------------------------------
    # CAUSAL MASK TEST
    # --------------------------------------------------------

    print("\nCausal-mask test:")

    try:
        sequence_length = 16

        base_x = make_test_input(
            vocab_size=EXPECTED_CONFIG["vocab_size"],
            batch_size=1,
            sequence_length=sequence_length,
        )

        changed_x = base_x.clone()

        # Change a future token.
        changed_position = 10

        old_token = changed_x[0, changed_position].item()

        new_token = (old_token + 1234) % EXPECTED_CONFIG["vocab_size"]

        if new_token == old_token:
            new_token = (old_token + 1) % EXPECTED_CONFIG["vocab_size"]

        changed_x[0, changed_position] = new_token

        with torch.no_grad():
            logits_base, _ = model(base_x)
            logits_changed, _ = model(changed_x)

        # Positions strictly before the changed future token
        # must remain identical.
        earlier_diff = (
            logits_base[:, :changed_position, :]
            - logits_changed[:, :changed_position, :]
        ).abs().max().item()

        # At/after the changed position, differences are allowed.
        later_diff = (
            logits_base[:, changed_position:, :]
            - logits_changed[:, changed_position:, :]
        ).abs().max().item()

        causal_pass = earlier_diff < 1e-6

        check(
            causal_pass,
            "Future token cannot affect earlier logits",
            f"earlier_max_diff={earlier_diff:.12f}",
        )

        check(
            later_diff > 1e-8,
            "Changed future position can affect itself/later logits",
            f"at_or_after_max_diff={later_diff:.12f}",
        )

    except Exception as e:
        check(
            False,
            "Causal mask test",
            repr(e),
        )

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("FINAL AUDIT SUMMARY")
print("=" * 75)

print(f"Total tests : {total_tests}")
print(f"Passed      : {passed_tests}")
print(f"Failed      : {failed_tests}")

if failed_tests == 0:
    print("\nOVERALL RESULT: PASS")
    print("Both Phase 2 models passed the final sanity audit.")
else:
    print("\nOVERALL RESULT: FAIL")
    print("At least one audit check failed. Review the [FAIL] lines above.")

print("=" * 75)