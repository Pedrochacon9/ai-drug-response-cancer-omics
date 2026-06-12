# Summary for the Meeting

## 1. Reorganization of the Approach

The project is no longer presented only as a table of results, but as a complete analysis of the pipeline:

1. understanding and preprocessing the data;
2. comparing models;
3. analyzing errors and interpreting the model behavior.

## 2. Explanatory Dataset Notebook

The following notebook has been created:

`notebooks_tfg/01_entender_dataset.ipynb`

This notebook explains:

* where the data comes from;
* what a cell line is;
* what a drug is;
* what a SMILES representation is;
* what gene expression is;
* what AUC means;
* how a training sample is built.

It also shows that the raw gene expression file contains initial metadata rows, which makes preprocessing an important step.

## 3. Drug-Out Analysis

The drug-out experiment has been analyzed beyond the global metric.

Main files:

* `tfg-notas/analisis_drugout/resumen_interpretativo_drugout.md`
* `tfg-notas/figuras/drugout_rmse_por_farmaco.png`
* `tfg-notas/figuras/drugout_scatter_real_predicho.png`
* `tfg-notas/figuras/drugout_error_absoluto_hist.png`

Main conclusion:

The error is not equally distributed across all drugs. `Drug_490` concentrates a large amount of error. In addition, the scatter plot shows horizontal bands, suggesting that in the drug-out setting the model tends to predict certain AUC levels and does not fully capture the real variability of some unseen drugs.

## 4. ECFP4 and Classical Baseline

An alternative drug representation has been generated:

`tfg-notas/experimentos_baselines/drug_ecfp.tsv`

Result:

* 1565 drugs converted;
* 2048 ECFP4 bits per drug;
* 0 invalid SMILES.

A Random Forest model has also been trained using:

`gene expression + ECFP4`

Main files:

* `tfg-notas/experimentos_baselines/rf_ecfp_pair/test_scores.json`
* `tfg-notas/experimentos_baselines/rf_ecfp_pair/summary.json`

Test result:

* RMSE: 0.0844
* PCC: 0.8593
* R2: 0.7381

Comparison:

The DeepTTC baseline achieves an RMSE of 0.0818, so Random Forest performs close to DeepTTC, although slightly worse.

## 5. Latent Vectors

`Step3_model.py` has been inspected, and clear points for extracting latent vectors have been identified.

Main files:

* `tfg-notas/analisis_latentes/localizacion_latentes.txt`
* `tfg-notas/analisis_latentes/resumen_localizacion_latentes.md`

Identified points:

1. `encoded_layers[:, 0]` can be interpreted as the drug latent vector.
2. `v_f = torch.cat((v_D, v_P), 1)` can be interpreted as the fused cell–drug latent vector.

The most interesting vector would be `fusion_latent = v_f`, because it represents the cell–drug pair just before the final AUC prediction.

## 6. Next Steps

The natural next steps would be:

1. train Random Forest or XGBoost also in the drug-out setting;
2. implement a controlled `return_latent=True` option in DeepTTC;
3. extract `fusion_latent` and analyze it with PCA/t-SNE;
4. study whether problematic drugs such as `Drug_490` appear separated in the latent space;
5. repeat some experiments with different seeds to check stability.
