# Model Architecture (DeepTTC): Drug Branch + Attention

## Overview

DeepTTC predicts drug response, by default **AUC**, by combining **two branches**:

1. **Drug branch (drug encoder)**: converts the SMILES representation into a vector using a **Transformer** with **multi-head self-attention**.
2. **Cell branch (gene encoder)**: processes **gene expression** through an **MLP** and produces another vector.

Both vectors are **concatenated** and passed through a **final MLP**, which outputs a single regression prediction.

**High-level pipeline**

* `SMILES → tokenization/ESPF → Transformer → z_drug`
* `genes → scaling → MLP → z_cell`
* `concat(z_drug, z_cell) → MLP → ŷ(AUC)`

---

## Drug Branch: SMILES → Embeddings → Transformer Attention

### 1. SMILES → Input Representation

During preprocessing, each molecule is represented in **SMILES** format. It is tokenized using `Step2_DataEncoding.py`, through the internal method `obj._drug2emb_encoder`, and stored as `drug_encoding`.

In the model `forward` pass, the drug input is handled as:

* `e = v[0]`: **tokens** as integer IDs
* `e_mask = v[1]`: **mask**, where `1` indicates a real token and `0` indicates padding

### 2. Embedding Layer

The tokens are projected into a dense vector space using:

* `input_dim_drug`: size of the token vocabulary, for example `2586`
* `transformer_emb_size_drug`: embedding dimension, for example `128`

This converts a sequence of token IDs into a sequence of dense vectors.

### 3. Attention Mechanism: Transformer Encoder

The main component of the drug branch is implemented in `Encoder_MultipleLayers(...)`, which defines a Transformer encoder with:

* `transformer_n_layer_drug`: number of encoder layers, for example `8`
* `transformer_num_attention_heads_drug`: number of attention heads, for example `8`
* `transformer_intermediate_size_drug`: size of the internal feed-forward network, for example `512`
* `transformer_attention_probs_dropout`: dropout applied to attention probabilities
* `transformer_hidden_dropout_rate`: dropout applied to hidden representations

The mask is expanded and transformed into the standard Transformer attention mask, assigning very negative values to padded positions so that the attention mechanism ignores padding tokens.

### 4. Final Drug Representation

The Transformer outputs a sequence of embeddings, one per token.

The model uses the embedding at **position 0** as the global representation of the molecule:

```python
drug_embedding = encoded_layers[:, 0]
```

In practice, this vector acts as an aggregation representation, similar to a **CLS** token, even though the SMILES sequence does not explicitly include a special CLS token.

---

## Cell Branch: Gene Expression → MLP

Gene expression is provided as a numerical vector with dimension `gene_dim`.

In our experiments, the model reports **958 input genes**, corresponding to a LINCS-like subset. This vector is processed through an `MLP`, which maps the gene expression profile into a cell embedding, set by default to `256` dimensions.

---

## Fusion and Final Prediction

Inside `Classifier.forward(...)`:

1. `v_D = model_drug(v_D)` → drug embedding
2. `v_P = model_gene(v_P)` → cell embedding
3. `v_f = concat(v_D, v_P)` → concatenation of both embeddings
4. Final MLP with hidden layers `(1024, 1024, 512)` → scalar output

The final output is a continuous regression prediction. The model is trained using **MSE loss**, and test performance is reported using metrics such as **MSE/RMSE**, **Pearson/Spearman correlation**, and **R²** to compare different runs.

---

## Key Parameters to Report for the Drug Branch

* `input_dim_drug`
* `transformer_emb_size_drug`
* `transformer_n_layer_drug`
* `transformer_num_attention_heads_drug`
* `transformer_intermediate_size_drug`
* `transformer_attention_probs_dropout`
* `transformer_hidden_dropout_rate`
* `dropout`

---

## Ablation Study: Drug Transformer Configuration

To analyze how sensitive the model is to the Transformer configuration used to process drugs, namely `SMILES → embeddings → attention`, two ablation experiments were performed on the TEST set:

* **heads4**: reduction of the number of attention heads
  `transformer_num_attention_heads_drug: 8 → 4`

* **layers4**: reduction of the number of encoder layers
  `transformer_n_layer_drug: 8 → 4`

The results show that the **baseline configuration with 8 attention heads and 8 encoder layers** achieves the best overall performance. Reducing the number of heads to 4 barely changes the metrics, although it does not improve performance. In contrast, reducing the number of layers to 4 leads to a clearer degradation, increasing RMSE and decreasing R².

This suggests that the **depth of the encoder** provides relevant representational capacity for capturing substructure-response relationships.

---

## Generated Documentation Material

* TEST figures: `tfg-notas/figuras/test_scatter_{baseline,heads4,layers4}.png`
* TEST summary table: `tfg-notas/tabla_resultados.md`
