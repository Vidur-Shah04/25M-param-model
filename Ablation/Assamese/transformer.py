import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .config import ModelConfig
except ImportError:
    from config import ModelConfig

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        # Create Q, K, V from the same input X.
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)

        # Combine the attention heads back into d_model.
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # True = future position, therefore cannot be attended to.
        mask = torch.triu(
            torch.ones(
                config.context_length,
                config.context_length,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        self.register_buffer("causal_mask", mask)

    def forward(self, x, return_attention=False):
        B, T, C = x.shape

        # X -> Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # [B, T, C] -> [B, heads, T, head_dim]
        q = q.view(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)

        # Scaled dot-product attention.
        scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # Prevent each position from seeing future tokens.
        scores = scores.masked_fill(
            self.causal_mask[:T, :T],
            float("-inf"),
        )

        # Convert attention scores into probabilities.
        attention = F.softmax(scores, dim=-1)

        # Save the attention probabilities BEFORE dropout.
        # This is important for analysis because these values
        # represent the actual normalized attention distribution.
        attention_for_analysis = attention

        # Dropout is used during training.
        attention = self.attn_dropout(attention)

        # Weighted combination of V.
        out = attention @ v

        # [B, heads, T, head_dim] -> [B, T, C]
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        out = self.out_proj(out)
        out = self.resid_dropout(out)

        if return_attention:
            return out, attention_for_analysis

        return out


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)

        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x, return_attention=False):
        if return_attention:
            attention_output, attention_weights = self.attention(
                self.ln1(x),
                return_attention=True,
            )

            # Pre-LayerNorm + residual connection.
            x = x + attention_output

            # Pre-LayerNorm + residual connection.
            x = x + self.ffn(self.ln2(x))

            return x, attention_weights

        # Normal training/inference path.
        # This is unchanged from the original implementation.
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        # Token embeddings: token ID -> vector.
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )

        # Learned positional embeddings.
        self.position_embedding = nn.Embedding(
            config.context_length,
            config.d_model,
        )

        self.embedding_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(config)
                for _ in range(config.n_layers)
            ]
        )

        self.final_ln = nn.LayerNorm(config.d_model)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02,
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):
                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02,
                )

    def forward(
        self,
        input_ids,
        targets=None,
        return_attention=False,
    ):
        B, T = input_ids.shape

        if T > self.config.context_length:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"context length {self.config.context_length}"
            )

        # Token positions: 0, 1, 2, ..., T-1
        positions = torch.arange(
            T,
            device=input_ids.device,
        )

        # Token + positional information.
        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
        )

        x = self.embedding_dropout(x)

        attention_maps = []

        # Pass through Transformer blocks.
        for block in self.blocks:

            if return_attention:
                x, attention_weights = block(
                    x,
                    return_attention=True,
                )

                attention_maps.append(attention_weights)

            else:
                x = block(x)

        x = self.final_ln(x)

        # Tied output projection.
        logits = F.linear(
            x,
            self.token_embedding.weight,
        )

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                targets.view(-1),
            )

        if return_attention:
            return logits, loss, attention_maps

        return logits, loss

    def count_parameters(self):
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )