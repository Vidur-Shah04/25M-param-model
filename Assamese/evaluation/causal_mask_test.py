from pathlib import Path
import sys

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from model.config import ModelConfig
from model.transformer import DecoderOnlyTransformer


CHECKPOINT_PATH = (
    PROJECT_DIR
    / "checkpoints"
    / "checkpoint_030518_assamese.pt"
)

SEQUENCE_LENGTH = 16
CHANGE_POSITION = 10


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("assamese CAUSAL MASK VERIFICATION")
print("=" * 70)

print(f"Device: {device}")


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=device,
    weights_only=False,
)

config = ModelConfig(**checkpoint["config"])

model = DecoderOnlyTransformer(config).to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# CREATE TWO INPUTS
# ============================================================

torch.manual_seed(42)

original = torch.randint(
    0,
    config.vocab_size,
    (1, SEQUENCE_LENGTH),
    device=device,
)

modified = original.clone()

# Change a future token.
modified[0, CHANGE_POSITION] = (
    modified[0, CHANGE_POSITION] + 1
) % config.vocab_size


print(
    f"Sequence length: {SEQUENCE_LENGTH}"
)

print(
    f"Changed future position: {CHANGE_POSITION}"
)


# ============================================================
# FORWARD PASSES
# ============================================================

with torch.no_grad():

    logits_original, _ = model(
        original
    )

    logits_modified, _ = model(
        modified
    )


# ============================================================
# CHECK POSITIONS BEFORE THE CHANGED TOKEN
# ============================================================

earlier_logits_original = (
    logits_original[:, :CHANGE_POSITION, :]
)

earlier_logits_modified = (
    logits_modified[:, :CHANGE_POSITION, :]
)

max_difference = torch.max(
    torch.abs(
        earlier_logits_original
        - earlier_logits_modified
    )
).item()


print()
print(
    "Maximum difference in earlier logits:"
)

print(
    f"{max_difference:.12f}"
)


# ============================================================
# CHECK FUTURE POSITION
# ============================================================

future_difference = torch.max(
    torch.abs(
        logits_original[:, CHANGE_POSITION:, :]
        - logits_modified[:, CHANGE_POSITION:, :]
    )
).item()


print(
    "Maximum difference at/after changed position:"
)

print(
    f"{future_difference:.12f}"
)


# ============================================================
# RESULT
# ============================================================

print()

if max_difference < 1e-7:

    print(
        "PASS: Future token does not affect "
        "earlier logits."
    )

else:

    print(
        "FAIL: Future token affected "
        "earlier logits."
    )

    raise RuntimeError(
        "Causal masking verification failed."
    )


print()
print("=" * 70)
print("CAUSAL MASK VERIFICATION COMPLETE")
print("=" * 70)