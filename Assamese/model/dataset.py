from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import IterableDataset


class TokenStreamDataset(IterableDataset):
    """
    Streams text shards and produces non-overlapping
    next-token prediction sequences.

    For seq_len = 512:

        tokens:
        [t0, t1, ..., t511]

        input:
        [t0, t1, ..., t510]

        target:
        [t1, t2, ..., t511]

    Therefore:
        x.shape = [511]
        y.shape = [511]
    """

    def __init__(
        self,
        data_dir,
        tokenizer_path,
        seq_len=512,
        split="train",
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.tokenizer_path = Path(tokenizer_path)
        self.seq_len = seq_len
        self.split = split

        # --------------------------------------------------
        # Load SentencePiece tokenizer
        # --------------------------------------------------

        self.sp = spm.SentencePieceProcessor()

        if not self.sp.load(str(self.tokenizer_path)):
            raise RuntimeError(
                f"Could not load tokenizer: "
                f"{self.tokenizer_path}"
            )

        # --------------------------------------------------
        # Find text shards
        # --------------------------------------------------

        self.files = sorted(
            self.data_dir.glob(
                f"{self.split}_*.txt"
            )
        )

        if not self.files:
            raise FileNotFoundError(
                f"No {self.split}_*.txt files found "
                f"in {self.data_dir}"
            )

        print(
            f"{self.split}: "
            f"found {len(self.files)} shards"
        )

    def __iter__(self):

        token_buffer = []

        for file_path in self.files:

            print(
                f"Reading {self.split} shard: "
                f"{file_path.name}"
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

                    # ------------------------------------------
                    # SentencePiece tokenization
                    # ------------------------------------------

                    ids = self.sp.encode(
                        line,
                        out_type=int,
                    )

                    token_buffer.extend(ids)

                    # ------------------------------------------
                    # Create non-overlapping 512-token chunks
                    # ------------------------------------------

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

        # Ignore incomplete final chunk.
        return