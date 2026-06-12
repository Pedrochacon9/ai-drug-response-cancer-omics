## Results (DeepTTC + IMPROVE)

DeepTTC was trained to predict drug response using the IMPROVE benchmark, combining gene expression data with a SMILES-based drug representation. The evaluation was performed on the TEST set, and regression metrics were reported, including MSE, RMSE, Pearson/Spearman correlation, and R². In addition, scatter plots comparing true AUC vs predicted AUC were generated to visually document prediction quality.

In the ablation study of the drug Transformer block, the **baseline configuration with 8 attention heads and 8 encoder layers** achieved the best overall performance. Reducing the number of attention heads to 4 (`heads4`) barely changed the results but did not provide any improvement. In contrast, reducing the number of encoder layers to 4 (`layers4`) produced a clearer performance degradation, increasing RMSE and decreasing R². This suggests that encoder depth contributes to capturing relevant relationships between drug structure and response.

### Limitations and Future Work

In the current split, train/validation/test share the same 24 drugs and part of the cell lines. Therefore, the evaluation mainly reflects generalization to **new cell–drug pairs**, rather than to completely unseen entities.

As an extension, more demanding generalization settings are proposed:

1. **Leave-drug-out split**: the model predicts responses for drugs that were not seen during training.
2. **Leave-cell-out split**: the model predicts responses for cell lines that were not seen during training.

These variants allow a better assessment of model robustness in discovery and translational scenarios.
