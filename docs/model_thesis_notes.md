## Model Description (DeepTTC)

DeepTTC is a drug response prediction model that combines information from the **drug** in SMILES format and the **cell line** through gene expression to predict a continuous response variable, in our case **AUC**. The central idea is to learn a numerical representation of the drug from SMILES using a **Transformer-based** block, and to learn a compact representation of the cell line from the gene expression vector using an **MLP**.

### Drug Branch (SMILES → Transformer)

The SMILES representation is first converted into a sequence of tokens, corresponding to molecular substructures, and then projected into embeddings. This sequence is processed by a **Transformer encoder** composed of multi-head self-attention and feed-forward layers, producing a summary vector for the drug.

In our experiments, the most relevant hyperparameters of this branch are the number of attention heads, `transformer_num_attention_heads_drug`, and the number of encoder layers, `transformer_n_layer_drug`, which were analyzed through ablation experiments.

### Cell Branch (Gene Expression → MLP)

Gene expression is represented as a fixed-size numerical vector. In our case, **958 genes** are used as input. This vector is passed through an **MLP** with several dense layers and ReLU activation functions to obtain a latent representation of the cell line.

This latent vector is intended to capture sensitivity and resistance patterns associated with the transcriptomic profile of the cell line.

### Fusion and Output (Regression)

The drug and cell latent vectors are **concatenated** and passed through a final MLP, referred to as the “classifier” in the code, although in this project it performs regression. This final block produces a single real-valued output: the predicted **AUC**.

Training is performed by minimizing **MSE**, and the model is evaluated on the TEST set using regression metrics such as **MSE/RMSE**, correlation metrics such as **Pearson/Spearman**, and **R²**.
