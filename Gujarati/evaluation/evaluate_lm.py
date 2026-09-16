from pathlib import Path
import sys
import math
import time

import sentencepiece as spm
import torch
from torch.utils.data import IterableDataset, DataLoader


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "checkpoint_030518_gujarati.pt"
)

TOKENIZER_PATH = (
    PROJECT_ROOT
    / "tokenizer_clean"
    / "gujarati_10k.model"
)

VAL_DIR = PROJECT_ROOT / "data" / "splits_v2"


sys.path.insert(0, str(MODEL_DIR))

from config import ModelConfig
from transformer import DecoderOnlyTransformer


# ---------------------------------------------------------
# Evaluation settings
# ---------------------------------------------------------

SEQ_LEN = 512
BATCH_SIZE = 1
PRINT_EVERY = 100

# Hard validation limit
MAX_VALIDATION_TOKENS = 100_000_000


# ---------------------------------------------------------
# Validation dataset
# ---------------------------------------------------------

class ValidationTokenDataset(IterableDataset):

    def __init__(
        self,
        data_dir,
        tokenizer_path,
        seq_len=512,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.seq_len = seq_len

        self.sp = spm.SentencePieceProcessor()

        if not self.sp.load(str(tokenizer_path)):
            raise RuntimeError(
                f"Could not load tokenizer: {tokenizer_path}"
            )

        self.files = sorted(
            self.data_dir.glob("val_*.txt")
        )

        if not self.files:
            raise FileNotFoundError(
                f"No validation files found in {self.data_dir}"
            )

    def __iter__(self):

        token_buffer = []

        for file_path in self.files:

            print(
                f"\nReading validation file: {file_path.name}"
            )

            with open(
                file_path,
                "r",
                encoding="utf-8",
            ) as f:

                for line in f:

                    line = line.strip()

                    if not line:
                        continue

                    ids = self.sp.encode(
                        line,
                        out_type=int,
                    )

                    token_buffer.extend(ids)

                    while len(token_buffer) >= self.seq_len:

                        chunk = token_buffer[
                            :self.seq_len
                        ]

                        del token_buffer[
                            :self.seq_len
                        ]

                        x = torch.tensor(
                            chunk[:-1],
                            dtype=torch.long,
                        )

                        y = torch.tensor(
                            chunk[1:],
                            dtype=torch.long,
                        )

                        yield x, y


# ---------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("GUJARATI PHASE 2 VALIDATION EVALUATION")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print(
        f"\nCheckpoint:\n{CHECKPOINT_PATH}"
    )

    print(
        f"Tokenizer:\n{TOKENIZER_PATH}"
    )

    print(
        f"Validation directory:\n{VAL_DIR}"
    )

    # -----------------------------------------------------
    # Load checkpoint
    # -----------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    # config = ModelConfig()
    config = ModelConfig(**checkpoint["config"])

    model = DecoderOnlyTransformer(
        config
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    # -----------------------------------------------------
    # Verify vocabulary
    # -----------------------------------------------------

    sp = spm.SentencePieceProcessor()

    if not sp.load(str(TOKENIZER_PATH)):
        raise RuntimeError(
            "Could not load Gujarati tokenizer."
        )

    tokenizer_vocab_size = sp.get_piece_size()

    print(
        f"\nTokenizer vocabulary size: "
        f"{tokenizer_vocab_size}"
    )

    print(
        f"Model vocabulary size: "
        f"{config.vocab_size}"
    )

    if tokenizer_vocab_size != config.vocab_size:
        raise ValueError(
            "Tokenizer vocabulary and model vocabulary "
            "do not match."
        )

    print(
        "PASS: tokenizer/model vocabulary match."
    )

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    dataset = ValidationTokenDataset(
        data_dir=VAL_DIR,
        tokenizer_path=TOKENIZER_PATH,
        seq_len=SEQ_LEN,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
    )

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    start_time = time.time()

    print("\nStarting validation...\n")

    with torch.no_grad():

        for x, y in loader:

            # Hard stop at 100M validation tokens
            remaining_tokens = (
                MAX_VALIDATION_TOKENS - total_tokens
            )

            if remaining_tokens <= 0:
                break

            x = x.to(device)
            y = y.to(device)

            # If this batch would exceed the limit,
            # keep only the required number of tokens.
            if y.numel() > remaining_tokens:
                tokens_to_keep = remaining_tokens

                x = x[:, :tokens_to_keep]
                y = y[:, :tokens_to_keep]

            logits, loss = model(
                x,
                y,
            )

            batch_tokens = y.numel()

            total_loss += (
                loss.item() * batch_tokens
            )

            total_tokens += batch_tokens
            num_batches += 1

            if num_batches % PRINT_EVERY == 0:

                current_ce = (
                    total_loss / total_tokens
                )

                current_ppl = math.exp(
                    current_ce
                )

                elapsed = (
                    time.time() - start_time
                )

                print(
                    f"Batch {num_batches:6d} | "
                    f"Tokens {total_tokens:,} | "
                    f"CE {current_ce:.6f} | "
                    f"PPL {current_ppl:.4f} | "
                    f"Time {elapsed/60:.2f} min"
                )
    # -----------------------------------------------------
    # Final metrics
    # -----------------------------------------------------

    final_ce = (
        total_loss / total_tokens
    )

    final_ppl = math.exp(
        final_ce
    )

    elapsed = (
        time.time() - start_time
    )

    print("\n" + "=" * 70)
    print("FINAL VALIDATION RESULTS")
    print("=" * 70)

    print(
        f"Validation sequences : {num_batches:,}"
    )

    print(
        f"Validation tokens    : {total_tokens:,}"
    )

    print(
        f"Cross-Entropy        : {final_ce:.6f}"
    )

    print(
        f"Perplexity           : {final_ppl:.4f}"
    )

    print(
        f"Evaluation time      : {elapsed/60:.2f} minutes"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
    