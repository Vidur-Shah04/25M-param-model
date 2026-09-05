from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 24_000
    d_model: int = 512
    n_layers: int = 5
    n_heads: int = 8
    d_ff: int = 1536
    context_length: int = 512
    dropout: float = 0.1

    @property
    def head_dim(self):
        return self.d_model // self.n_heads