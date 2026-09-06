from pathlib import Path
import sys
import torch

# Allow imports from Gujarati/model/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
sys.path.insert(0, str(MODEL_DIR))

from config import ModelConfig
from transformer import DecoderOnlyTransformer


CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "checkpoint_030518_gujarati.pt"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("GUJARATI PHASE 2 CHECKPOINT VERIFICATION")
    print("=" * 60)

    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # Load checkpoint
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    print("\nCheckpoint keys:")
    for key in checkpoint.keys():
        print(f"  - {key}")

    # Load configuration
    config = ModelConfig()

    print("\nModel configuration:")
    print(f"  vocab_size      = {config.vocab_size}")
    print(f"  d_model         = {config.d_model}")
    print(f"  n_layers        = {config.n_layers}")
    print(f"  n_heads         = {config.n_heads}")
    print(f"  d_ff            = {config.d_ff}")
    print(f"  context_length  = {config.context_length}")
    print(f"  dropout         = {config.dropout}")
    print(f"  head_dim        = {config.head_dim}")

    # Build model
    model = DecoderOnlyTransformer(config).to(device)

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    # Parameter count
    total_params = count_parameters(model)

    print("\nParameter count:")
    print(f"  {total_params:,}")

    expected_params = 25_689_088

    if total_params == expected_params:
        print("  PASS: parameter count matches expected architecture.")
    else:
        print(
            f"  WARNING: expected {expected_params:,}, "
            f"but found {total_params:,}."
        )

    # Checkpoint training information
    print("\nCheckpoint training information:")

    for key in [
        "step",
        "loss",
        "lr",
        "config",
    ]:
        if key in checkpoint:
            print(f"  {key}: {checkpoint[key]}")

    # Forward-pass test
    batch_size = 1
    seq_len = 32

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (batch_size, seq_len),
        device=device,
    )

    targets = torch.randint(
        0,
        config.vocab_size,
        (batch_size, seq_len),
        device=device,
    )

    with torch.no_grad():
        logits, loss = model(input_ids, targets)

    print("\nForward-pass test:")
    print(f"  Input shape:  {tuple(input_ids.shape)}")
    print(f"  Logits shape: {tuple(logits.shape)}")
    print(f"  Loss:         {loss.item():.6f}")

    expected_shape = (
        batch_size,
        seq_len,
        config.vocab_size,
    )

    if tuple(logits.shape) == expected_shape:
        print("  PASS: logits shape is correct.")
    else:
        print(
            f"  FAIL: expected {expected_shape}, "
            f"got {tuple(logits.shape)}."
        )

    if torch.isfinite(loss):
        print("  PASS: loss is finite.")
    else:
        print("  FAIL: loss is not finite.")

    print("\n" + "=" * 60)
    print("CHECKPOINT VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
    