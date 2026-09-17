
from dataclasses import dataclass


@dataclass
class ModelConfig:

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    vocab_size: int = 10_000

    # --------------------------------------------------------
    # Transformer dimensions
    # --------------------------------------------------------

    d_model: int = 512

    n_layers: int = 7

    n_heads: int = 8

    d_ff: int = 1792

    # --------------------------------------------------------
    # Context
    # --------------------------------------------------------

    context_length: int = 512

    # --------------------------------------------------------
    # Regularization
    # --------------------------------------------------------

    dropout: float = 0.1

    @property
    def head_dim(self):

        return self.d_model // self.n_heads
