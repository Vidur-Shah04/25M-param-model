from pathlib import Path
import math
import time

import torch

from config import ModelConfig
from transformer import DecoderOnlyTransformer
from dataset import TokenStreamDataset


# ============================================================
# PATHS
# ============================================================

# Gujarati/
# ├── model/
# ├── tokenizer_clean/
# └── train/

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_DIR
    / "train"
)

TOKENIZER_PATH = (
    PROJECT_DIR
    / "tokenizer_clean"
    / "gujarati_24k.model"
)

CHECKPOINT_DIR = (
    PROJECT_DIR
    / "train"
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 1

GRAD_ACCUMULATION = 32

LEARNING_RATE = 3e-4

MIN_LR = 3e-5

WEIGHT_DECAY = 0.1

BETAS = (0.9, 0.95)

TARGET_TOKENS = 500_000_000

WARMUP_STEPS = 2_000

SAVE_EVERY = 1_500

VALIDATE_EVERY = 1_500

VAL_BATCHES = 100

GRAD_CLIP = 1.0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

if device.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )


# ============================================================
# MODEL
# ============================================================

config = ModelConfig()

model = DecoderOnlyTransformer(
    config
).to(device)

print(
    "Parameters:",
    model.count_parameters(),
)


# ============================================================
# TRAINING STEPS
# ============================================================

TOKENS_PER_STEP = (
    BATCH_SIZE
    * GRAD_ACCUMULATION
    * config.context_length
)

MAX_STEPS = math.ceil(
    TARGET_TOKENS
    / TOKENS_PER_STEP
)

print(
    "Tokens per optimizer step:",
    TOKENS_PER_STEP,
)

print(
    "Target tokens:",
    TARGET_TOKENS,
)

print(
    "Maximum steps:",
    MAX_STEPS,
)


# ============================================================
# DATASET
# ============================================================

train_dataset = TokenStreamDataset(
    data_dir=DATA_DIR,
    tokenizer_path=TOKENIZER_PATH,
    seq_len=config.context_length,
    split="train",
)

train_iterator = iter(
    train_dataset
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=BETAS,
    weight_decay=WEIGHT_DECAY,
)


# ============================================================
# LEARNING RATE
# ============================================================

def get_lr(step):

    # Linear warmup
    if step < WARMUP_STEPS:

        return (
            LEARNING_RATE
            * (step + 1)
            / WARMUP_STEPS
        )

    # Cosine decay
    progress = (
        step - WARMUP_STEPS
    ) / (
        MAX_STEPS - WARMUP_STEPS
    )

    progress = min(
        progress,
        1.0,
    )

    return (
        MIN_LR
        + (
            LEARNING_RATE
            - MIN_LR
        )
        * 0.5
        * (
            1.0
            + math.cos(
                math.pi * progress
            )
        )
    )


# ============================================================
# MIXED PRECISION
# ============================================================

use_amp = (
    device.type == "cuda"
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp,
)


# ============================================================
# CHECKPOINT
# ============================================================

last_loss = 0.0


def save_checkpoint(step):

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"checkpoint_{step:06d}.pt"
    )

    checkpoint = {

        "step": step,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "scaler_state_dict":
            scaler.state_dict(),

        "config":
            vars(config),

        "loss":
            last_loss,

        "learning_rate":
            optimizer.param_groups[0]["lr"],

        "target_tokens":
            TARGET_TOKENS,

        "tokens_per_step":
            TOKENS_PER_STEP,
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )

    print(
        f"Checkpoint saved: "
        f"{checkpoint_path}"
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(path):

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    if "scaler_state_dict" in checkpoint:

        scaler.load_state_dict(
            checkpoint[
                "scaler_state_dict"
            ]
        )

    step = checkpoint["step"]

    print(
        f"Resumed from step {step}"
    )

    return step


# ============================================================
# RESUME
# ============================================================

RESUME_CHECKPOINT = None

# Example:
#
# RESUME_CHECKPOINT = (
#     CHECKPOINT_DIR
#     / "checkpoint_015000.pt"
# )

start_step = 0

if RESUME_CHECKPOINT is not None:

    start_step = load_checkpoint(
        RESUME_CHECKPOINT
    )


# ============================================================
# VALIDATION
# ============================================================

def validate():

    model.eval()

    val_dataset = TokenStreamDataset(
        data_dir=DATA_DIR,
        tokenizer_path=TOKENIZER_PATH,
        seq_len=config.context_length,
        split="val",
    )

    val_iterator = iter(
        val_dataset
    )

    losses = []

    with torch.no_grad():

        for _ in range(
            VAL_BATCHES
        ):

            x, y = next(
                val_iterator
            )

            x = (
                x
                .unsqueeze(0)
                .to(device)
            )

            y = (
                y
                .unsqueeze(0)
                .to(device)
            )

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):

                _, loss = model(
                    x,
                    y,
                )

            losses.append(
                loss.item()
            )

    val_loss = (
        sum(losses)
        / len(losses)
    )

    perplexity = math.exp(
        val_loss
    )

    print(
        f"Validation loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation PPL: "
        f"{perplexity:.2f}"
    )

    model.train()

    return val_loss


# ============================================================
# TRAINING LOOP
# ============================================================

model.train()

start_time = time.time()

print(
    "\nStarting Gujarati training..."
)

for step in range(
    start_step + 1,
    MAX_STEPS + 1,
):

    optimizer.zero_grad(
        set_to_none=True
    )

    accumulated_loss = 0.0

    # --------------------------------------------------------
    # Gradient accumulation
    # --------------------------------------------------------

    for _ in range(
        GRAD_ACCUMULATION
    ):

        x, y = next(
            train_iterator
        )

        x = (
            x
            .unsqueeze(0)
            .to(device)
        )

        y = (
            y
            .unsqueeze(0)
            .to(device)
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):

            _, loss = model(
                x,
                y,
            )

            loss = (
                loss
                / GRAD_ACCUMULATION
            )

        scaler.scale(
            loss
        ).backward()

        accumulated_loss += (
            loss.item()
        )

    # --------------------------------------------------------
    # Gradient clipping
    # --------------------------------------------------------

    scaler.unscale_(
        optimizer
    )

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRAD_CLIP,
    )

    # --------------------------------------------------------
    # Learning rate
    # --------------------------------------------------------

    lr = get_lr(step)

    for group in optimizer.param_groups:

        group["lr"] = lr

    # --------------------------------------------------------
    # Optimizer step
    # --------------------------------------------------------

    scaler.step(
        optimizer
    )

    scaler.update()

    last_loss = accumulated_loss

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    if step % 100 == 0:

        elapsed = (
            time.time()
            - start_time
        )

        tokens_done = (
            step
            * TOKENS_PER_STEP
        )

        print(
            f"step={step} | "
            f"loss={last_loss:.4f} | "
            f"lr={lr:.6e} | "
            f"tokens={tokens_done:,} | "
            f"time={elapsed:.1f}s"
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if (
        step % VALIDATE_EVERY
        == 0
    ):

        validate()

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    if (
        step % SAVE_EVERY
        == 0
    ):

        save_checkpoint(step)


# ============================================================
# FINAL CHECKPOINT
# ============================================================

if (
    MAX_STEPS % SAVE_EVERY
    != 0
):

    save_checkpoint(
        MAX_STEPS
    )


print(
    "\nGujarati training complete."
)