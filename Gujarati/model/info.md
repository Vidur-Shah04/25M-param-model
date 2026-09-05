24K vocab · 512 hidden · 8 heads · 5 blocks · 1536 FFN · 512 context · 0.1 dropout · tied embeddings

8 heads × 64 dimensions/head = 512
| Component         |  Choice |
| ----------------- | ------: |
| Vocabulary        |  24,000 |
| `d_model`         |     512 |
| Layers            |       5 |
| Attention heads   |       8 |
| Head dimension    |      64 |
| FFN hidden size   |   1,536 |
| Context length    |     512 |
| Dropout           |     0.1 |
| Position encoding | Learned |
| Embeddings        |    Tied |

Total: 25,406,464 parameters ≈ 25.41M.