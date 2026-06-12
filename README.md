# AI-based Drug Response Prediction in Cancer Omics

This repository contains the code, analysis and results developed for my Bachelor's Thesis in Health Engineering, focused on **drug response prediction in cancer** using **deep learning** and **omics data**.

The project explores how artificial intelligence models can combine:

- Gene expression profiles from cancer cell lines
- Molecular representations of drugs
- Experimental drug response values

to predict the expected response of a tumor cell line to a given drug.

## Project overview

The main objective of this work is to evaluate the behaviour of **DeepTTC**, a deep learning model for drug response prediction, on the **IMPROVE/CSA benchmark**.

The work focuses not only on predictive performance, but also on model generalization under different evaluation scenarios:

- Normal split
- Cell-out split
- Drug-out split

The most relevant finding is that DeepTTC performs well in standard scenarios, but generalization to unseen drugs remains a major challenge.

## Research context

Drug response prediction is an active research area within artificial intelligence applied to cancer. The goal is not to replace clinical decision-making, but to study how computational models can learn patterns from biomedical data and support future research in precision medicine.

In this project, the model learns relationships between:

- The molecular profile of a cancer cell line
- The chemical structure of a drug
- The observed experimental response

## Dataset

The experiments are based on the **IMPROVE/CSA benchmark** for drug response prediction.

The key data sources used in the project are:

- `response.tsv`: experimental drug response values
- `cancer_gene_expression.tsv`: gene expression profiles
- `drug_SMILES.tsv`: molecular representation of drugs

The original full dataset is not included in this repository. Only small sample files are provided in `data_sample/` to show the expected data format.

## Model

The main model used is **DeepTTC**, a deep learning architecture that combines:

- A drug branch based on SMILES tokenization and Transformer-based processing
- A cell branch based on gene expression data
- A fusion module to predict the AUC response value

A classical baseline based on **Random Forest + ECFP4 molecular fingerprints** is also included for comparison.

## Evaluation

The model was evaluated using different generalization scenarios:

- **Normal split**: standard evaluation scenario.
- **Cell-out split**: the model is evaluated on unseen cancer cell lines.
- **Drug-out split**: the model is evaluated on unseen drugs.

This allows the project to analyse not only whether the model predicts well, but also under which conditions it generalizes better or worse.

## Main results

The results show that:

- DeepTTC achieves strong performance in the normal split.
- Cell-out generalization remains reasonably stable.
- Drug-out generalization is much more challenging.
- Some drugs concentrate significantly higher prediction errors.
- Random Forest + ECFP4 remains a competitive classical baseline.

The main conclusion is that deep learning models can perform well in standard settings, but generalization to completely unseen drugs remains an important open challenge.

## Repository structure

```text
notebooks/     Exploratory analysis and experiment notebooks
src/           Python scripts for preprocessing, training and evaluation
results/       Summary metrics and experiment outputs
figures/       Main figures generated for analysis
docs/          Thesis notes and documentation
data_sample/   Small sample files showing the input data format
```

# Key folders
**src/**

Contains the adapted DeepTTC pipeline and auxiliary scripts for preprocessing, training, inference and result processing.

**data_sample/**

Contains small sample files derived from the original dataset structure. These files are intended only to illustrate the expected input format.

**figures/**

Contains visualizations generated during the experimental analysis, including real vs predicted AUC plots and drug-out error analysis.

**results/**

Contains prediction outputs and experiment-related results. The baseline outputs are organized under results/baseline/.

**docs/**

Contains complementary documentation about the dataset, model, methodology, experimental design and results.

**Technologies used**
Python
pandas
NumPy
scikit-learn
PyTorch
Deep learning
Random Forest
Molecular fingerprints
SMILES representations
Omics data analysis
Git/GitHub
Limitations

**This project is research-oriented and has several important limitations:**

It uses cancer cell line data, not direct patient-level clinical data.
The model has not been clinically validated.
Drug-out generalization remains challenging.
The results should be interpreted as experimental and exploratory.
The repository is not intended for medical use or treatment recommendation.
Future work

**Possible future research directions include:**

Improving drug-out generalization.
Testing graph neural networks for molecular representation.
Using pretrained molecular models.
Combining SMILES and ECFP4 representations.
Adding multi-omics data such as mutations, copy number variation or methylation.
Adding uncertainty estimation and out-of-distribution detection.
Building a research-oriented platform for drug response hypothesis prioritization.
Product-oriented vision

This project can be extended into a research platform for oncology drug response analysis.

**A possible future product direction is:**

An AI-powered research platform that integrates omics data and molecular drug representations to prioritize drug response hypotheses in cancer, with explainability, uncertainty estimation and benchmark-based validation.

This should be positioned as a research and hypothesis prioritization tool, not as a clinical decision-making system.

# Author

Pedro Chacón Cabrera
Health Engineering — Clinical Informatics
University of Seville
