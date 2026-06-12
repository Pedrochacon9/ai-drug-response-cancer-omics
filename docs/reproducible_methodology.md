## Reproducible Methodology (DeepTTC + IMPROVE)

1. **Environment**: WSL Ubuntu with the conda environment `deepttc` and GPU available (`torch.cuda.is_available() = True`, `cuda:0`).

2. **Code**: main `DeepTTC` repository, which contains the model implementation, together with the `IMPROVE` repository installed in editable mode using `pip install -e .` in order to use the utilities from the IMPROVE standard.

3. **Data (IMPROVE/CSA)**: the following benchmark files are used:

* **Drug data**: `csa_data/raw_data/x_data/drug_SMILES.tsv` containing SMILES and/or derived molecular representations.
* **Omics data**: `csa_data/raw_data/x_data/cancer_gene_expression.tsv` containing gene expression data.
* **Response data**: `csa_data/raw_data/y_data/response.tsv` containing the target variable `auc`.

4. **Preprocessing validation**

The quality of the preprocessed train/validation/test datasets was checked by verifying:

* Absence of missing values, with `NaNs = 0` in all splits.
* Absence of duplicated `(cell line, drug)` pairs.
* No overlap of pairs between splits: `train∩val = 0`, `train∩test = 0`, `val∩test = 0`. This avoids data leakage at the sample level.

**Dataset size and overlap in the normal pair split**

* Train: 7616 samples, 411 cell lines, 24 drugs.
* Validation: 952 samples, 371 cell lines, 24 drugs.
* Test: 951 samples, 371 cell lines, 24 drugs.

The `(cell line, drug)` pairs are **fully disjoint** across train/validation/test.

However, there is overlap at the entity level: the same **24 drugs** appear in the three splits, and there is partial overlap of **cell lines**. Therefore, this split evaluates generalization to **new pairs**, rather than to completely unseen entities.

**Preprocessing output**

The preprocessing step generates HDF5 datasets:

* `exp_result/train_data.h5`
* `exp_result/val_data.h5`
* `exp_result/test_data.h5`

Expected HDF5 keys:

* `drug`
* `gene_expression`

5. **Training**

The model is trained and the execution artifacts are stored:

* Trained model: `model.pt`
* Logs: `train_*.log`
* Validation scores, when applicable: `val_scores.json` or metrics printed in the log.

6. **TEST inference**

Inference is executed on `test_data.h5`, and predictions and metrics are saved:

* Predictions: `test_y_data_predicted.csv`
* Metrics: `test_scores.json`
* True vs predicted table: `test_results.tsv`
* Scatter plot: `test_scatter.png`

The scatter plot is generated inside the run folder and copied to `tfg-notas/figuras/` for documentation.

7. **Experiments / ablations**

Each configuration is executed in an independent folder:

* `exp_result/runs/<run>/`

Examples:

* `baseline`
* `heads4`
* `layers4`
* `drugout_seed42`
* `cellout_seed42`

This keeps the execution traceability of:

* `model.pt`
* logs
* `test_scores.json`
* inference outputs

8. **Documentation**

The workflow is recorded in a logbook, and documentation-ready material is generated:

* `tfg-notas/bitacora.md`
* `tfg-notas/tabla_resultados.md`
* `tfg-notas/resultados_memoria.md`

**How to reproduce a run**

Enter the corresponding run folder:

`exp_result/runs/<run>/`

Then execute the general workflow:

1. training;
2. TEST inference;
3. copy the figure and scores to `tfg-notas/figuras/`.
