# Dataset / Data Documentation — DeepTTC + IMPROVE

We use the **IMPROVE benchmark for drug response prediction** as the standard data source. The **DeepTTC** repository consumes the data in an “IMPROVE-style” format: part of the data is already provided under `csa_data/raw_data/` as X and Y features, while another part comes from `author_data/`, mainly the vocabulary/ESPF resources required to encode SMILES.

In our workflow, we first run the **preprocessing step**, which generates `train_data.h5`, `val_data.h5`, `test_data.h5`, and auxiliary CSV files. Then, training and inference are performed using these prepared files.

## Main Repositories / Paths (WSL)

* Model repository: `~/tfg/DeepTTC/`
* IMPROVE repository/library: `~/tfg/IMPROVE/`
  Installed with `pip install -e .` so that `import improvelib` works correctly.

## Raw Data Files Used by DeepTTC

### Target Variable (Y)

* `csa_data/raw_data/y_data/response.tsv`
  Contains the response/label for each **(improve_sample_id, improve_chem_id)** pair.
  In our runs, the prediction target is `auc`.

### Drug Features (X-drug)

* `csa_data/raw_data/x_data/drug_SMILES.tsv`
  Maps **improve_chem_id → SMILES/canSMILES**.
  This is the input used by the drug Transformer branch.

* Alternative files available, although not required when using SMILES:

  * `csa_data/raw_data/x_data/drug_ecfp4_nbits512.tsv`
  * `csa_data/raw_data/x_data/drug_mordred.tsv`
  * `csa_data/raw_data/x_data/drug_info.tsv`

### Cell Line / Cancer Features (X-cell)

* `csa_data/raw_data/x_data/cancer_gene_expression.tsv`
  Gene expression matrix indexed by `improve_sample_id`.
  In our case, the model reports **958 input genes**.

### Train / Validation / Test Splits

* `CCLE_split_0_train.txt`
* `CCLE_split_0_val.txt`
* `CCLE_split_0_test.txt`

## Resources Required to Encode SMILES (`author_data`)

* `author_data/ESPF/`
  Vocabulary and subword mapping files used to tokenize SMILES with ESPF.

  * `author_data/ESPF/subword_units_map_uniprot_2000.csv`
  * `author_data/ESPF/subword_units_map_chembl_freq_1500.csv`

## Preprocessing Outputs Used for Training and Inference

The preprocessing outputs are stored in `exp_result/`, or inside `exp_result/runs/<run>/` when using separate experiment runs:

* `train_data.h5`
* `val_data.h5`
* `test_data.h5`
* `train_y_data.csv`
* `val_y_data.csv`
* `test_y_data.csv`

## Typical Training / Inference Outputs

* Trained model: `model.pt`
* Scores: `val_scores.json`, `test_scores.json`
* Predictions: `val_y_data_predicted.csv`, `test_y_data_predicted.csv`
* True vs predicted tables: `val_results.tsv`, `test_results.tsv`
* Scatter plots: `val_scatter.png`, `test_scatter.png`

---

## Selected Dataset Explanation

We use the **IMPROVE benchmark for Drug Response Prediction**. The benchmark originates from **IMPROVE/CANDLE (Argonne National Laboratory)**. In our project, these files are already organized under `csa_data/raw_data/`, and the pipeline preprocesses them into HDF5 files for training and inference.

The raw data consumed by DeepTTC, namely drug SMILES, cancer cell line gene expression, and AUC response values for each cell–drug pair, are stored under `csa_data/raw_data/`. DeepTTC is designed to work with this standard “IMPROVE-style” format.

In addition, to execute the workflow as expected by the framework, we install the **`improvelib`** library from the **IMPROVE** repository located at `~/tfg/IMPROVE/` using `pip install -e .`. This provides utilities and structure for training and inference under the IMPROVE standard.

* **Dataset files:** files under `csa_data/raw_data/`
  X features: `drug_SMILES.tsv`, `cancer_gene_expression.tsv`
  Y target: `response.tsv`
  Splits: `CCLE_split_0_*.txt`

* **Encoding resources:** `author_data/ESPF/`
  These are not part of the benchmark data itself, but they are required to tokenize SMILES.

* **Tooling / standard code:** `IMPROVE/` repository
  Installed as `improvelib` to run the workflow in the IMPROVE-style format.

## How to Reproduce

* Clone the DeepTTC repository into:

  `~/tfg/DeepTTC/`

  This repository includes the `csa_data/raw_data/` directory.

* Clone the IMPROVE repository into:

  `~/tfg/IMPROVE/`

* Install IMPROVE in editable mode:

  `pip install -e .`

  This allows Python to import `improvelib`.

* Run the preprocessing step to generate the `.h5` files and auxiliary CSV files in:

  `exp_result/`

  or, for separated runs:

  `exp_result/runs/<run>/`

## Notes on Data Splits

* **Normal split:** `exp_result/`
  Standard train/validation/test split with no overlapping cell–drug pairs between splits.

* **Generalization splits:**

  * `exp_result/runs/drugout_seed42/`
    Leave-drug-out setting, used to evaluate generalization to unseen drugs.

  * `exp_result/runs/cellout_seed42/`
    Leave-cell-out setting, used to evaluate generalization to unseen cell lines.
