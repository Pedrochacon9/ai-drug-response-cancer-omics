# Experimental Plan of the Final Degree Project

## General Objective

The objective of this work is not only to obtain a final prediction metric, but also to analyze how the input data, preprocessing, drug representation, and type of data split influence drug response prediction.

The studied problem consists of predicting the AUC value for a cell line–drug pair using:

* gene expression of the cell line;
* molecular representation of the drug;
* observed AUC response as the target variable.

## Work Blocks

The project is organized into three main blocks:

1. Understanding and preprocessing the dataset.
2. Experimental comparison of models.
3. Analysis and interpretation of the model behavior.

## Experiments Performed

| Experiment | Model           | Input                    | Split      | Objective                                    |
| ---------- | --------------- | ------------------------ | ---------- | -------------------------------------------- |
| E1         | DeepTTC         | SMILES + gene expression | pair split | Evaluate performance on new cell–drug pairs  |
| E2         | DeepTTC heads4  | SMILES + gene expression | pair split | Ablation of the number of attention heads    |
| E3         | DeepTTC layers4 | SMILES + gene expression | pair split | Ablation of the number of Transformer layers |
| E4         | DeepTTC         | SMILES + gene expression | cell-out   | Evaluate generalization to unseen cell lines |
| E5         | DeepTTC         | SMILES + gene expression | drug-out   | Evaluate generalization to unseen drugs      |

## Results Observed So Far

The results show that the model performs well on the normal cell–drug pair split and maintains reasonable performance in the cell-out setting.

However, performance drops significantly in the drug-out setting.

This suggests that generalization to completely unseen drugs is the most challenging part of the problem.

## Proposed Experiments

| Experiment | Model                          | Input                   | Split         | Objective                                                                |
| ---------- | ------------------------------ | ----------------------- | ------------- | ------------------------------------------------------------------------ |
| E6         | Random Forest                  | gene expression + ECFP4 | pair split    | Classical baseline                                                       |
| E7         | XGBoost                        | gene expression + ECFP4 | pair split    | Strong classical baseline                                                |
| E8         | Random Forest                  | gene expression + ECFP4 | drug-out      | Check whether the drop on unseen drugs also appears in a classical model |
| E9         | XGBoost                        | gene expression + ECFP4 | drug-out      | Compare drug-out generalization against DeepTTC                          |
| E10        | DeepTTC latent representations | internal vectors        | pair/drug-out | Analyze what the model learns internally                                 |

## Complementary Analyses

In addition to global metrics, several complementary analyses will be performed:

* AUC distribution;
* number of samples per drug;
* number of samples per cell line;
* entity overlap between splits;
* error per drug in the drug-out setting;
* comparison between true AUC and predicted AUC;
* extraction and visualization of latent vectors.

## Justification

The normal split evaluates new cell–drug pairs, but it does not evaluate completely unseen drugs because the same drugs appear in training, validation, and test sets.

For this reason, it is necessary to study more demanding splits, especially the drug-out setting.

The drug-out analysis allows us to evaluate whether the model learns a generalizable drug representation or whether it depends too heavily on having seen the drugs during training.

## Preliminary Random Forest Result with ECFP4

A first classical baseline was executed using Random Forest with:

`gene expression + ECFP4`

The final input contains 3006 features:

* 958 genes;
* 2048 ECFP4 bits.

Test result on the normal split:

| Model            | Input          | Split      |   RMSE |    PCC |     R2 |
| ---------------- | -------------- | ---------- | -----: | -----: | -----: |
| DeepTTC baseline | SMILES + genes | pair split | 0.0818 | 0.8685 | 0.7542 |
| Random Forest    | ECFP4 + genes  | pair split | 0.0844 | 0.8593 | 0.7381 |

This result shows that the classical baseline is close to DeepTTC on the normal split, although it performs slightly worse.
