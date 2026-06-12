## 2026-03-13 — TEST Inference (DeepTTC + IMPROVE)

**Environment**

* WSL Ubuntu + conda environment: `deepttc`
* GPU: `torch.cuda.is_available() = True` (`cuda:0`)

**Inputs used**

* Processed data: `exp_result/test_data.h5`
* Model: `exp_result/model.pt`

**TEST inference command**

```bash
cd ~/tfg/DeepTTC
python deepttc_infer_improve.py \
  --input_data_dir ./exp_result \
  --input_model_dir ./exp_result \
  --model_file_name model \
  --model_file_format .pt \
  --cuda_name cuda:0 \
  --calc_infer_scores True \
  -o ./exp_result
```

**Generated files**

* `exp_result/test_y_data_predicted.csv`
* `exp_result/test_scores.json`
* `exp_result/test_results.tsv`
* `exp_result/test_scatter.png`

**TEST metrics**

* MSE: 0.00962566596353434
* RMSE: 0.09811047835748402
* PCC (Pearson): 0.8140919644360205
* SCC (Spearman): 0.6676354143741668
* R2: 0.6464630026737932

---

## 2026-03-13 — Ablation: Number of Attention Heads (TEST)

**Objective:** evaluate the impact of reducing `transformer_num_attention_heads_drug` from 8 to 4.

**TEST results (`test_scores.json`):**

| Experiment         |        MSE |       RMSE |        PCC |        SCC |         R2 |
| ------------------ | ---------: | ---------: | ---------: | ---------: | ---------: |
| baseline (heads=8) | 0.00669110 | 0.08179915 | 0.86853303 | 0.76613500 | 0.75424541 |
| heads4 (heads=4)   | 0.00674747 | 0.08214296 | 0.86816928 | 0.76525122 | 0.75217517 |

**Run folders**

* `exp_result/runs/baseline/`
* `exp_result/runs/heads4/`

**Conclusion:** reducing the number of heads from 8 to 4 does not improve performance. It is slightly worse in terms of MSE, RMSE, and R2.

---

## 2026-03-13 — Ablation: Number of Drug Transformer Layers (TEST)

**Objective:** evaluate the impact of reducing `transformer_n_layer_drug` from 8 to 4.

**Result for `layers4` on TEST**

* MSE: 0.00695003
* RMSE: 0.08336682
* PCC (Pearson): 0.86561822
* SCC (Spearman): 0.76372329
* R2: 0.74473540

**Run folder**

* `exp_result/runs/layers4/`

**Conclusion:** reducing the number of layers from 8 to 4 slightly worsens performance compared with the baseline, with worse MSE, RMSE, PCC, SCC, and R2.

---

## 2026-03-13 — Final Ablation Comparison (TEST)

**Compared runs**

* `exp_result/runs/baseline/`  (heads=8, layers=8)
* `exp_result/runs/heads4/`    (heads=4, layers=8)
* `exp_result/runs/layers4/`   (heads=8, layers=4)

**TEST metrics**

| Run      | Heads | Layers |        MSE |       RMSE |        PCC |        SCC |         R2 |
| -------- | ----: | -----: | ---------: | ---------: | ---------: | ---------: | ---------: |
| baseline |     8 |      8 | 0.00669110 | 0.08179915 | 0.86853303 | 0.76613500 | 0.75424541 |
| heads4   |     4 |      8 | 0.00674747 | 0.08214296 | 0.86816928 | 0.76525122 | 0.75217517 |
| layers4  |     8 |      4 | 0.00695003 | 0.08336682 | 0.86561822 | 0.76372329 | 0.74473540 |

**Conclusion**

* The **baseline** configuration, with 8 attention heads and 8 Transformer layers, performs best.
* Reducing the number of heads to 4 barely changes the metrics, but it does **not improve** performance.
* Reducing the number of layers to 4 leads to a clearer degradation, increasing RMSE and decreasing R2.

**Figures for the report**

* `tfg-notas/figuras/test_scatter_baseline.png`
* `tfg-notas/figuras/test_scatter_heads4.png`
* `tfg-notas/figuras/test_scatter_layers4.png`

**Summary table for the report**

* `tfg-notas/tabla_resultados.md`

**Report summary**

* Table: `tfg-notas/tabla_resultados.md`
* Text: `tfg-notas/resultados_memoria.md`
* Figures: `tfg-notas/figuras/`

**Reproducible methodology**

* `tfg-notas/metodologia_reproducible.md`

**Model description for the report**

* `tfg-notas/modelo_memoria.md`

---

## 2026-03-16 — Ablation: Leave-Drug-Out (TEST) — `drugout_seed42`

**Objective:** evaluate generalization to **unseen drugs**. In this setting, some drugs are removed from training, and the TEST set contains those held-out drugs.

**Experiment folder**

* `exp_result/runs/drugout_seed42/`
* Held-out drug list: `exp_result/runs/drugout_seed42/heldout_drugs.txt`

**Generated files (TEST)**

* `exp_result/runs/drugout_seed42/test_scores.json`
* `exp_result/runs/drugout_seed42/test_y_data_predicted.csv`
* `exp_result/runs/drugout_seed42/test_results.tsv`
* `exp_result/runs/drugout_seed42/test_scatter.png`

**Copies stored in notes**

* `tfg-notas/figuras/test_scatter_drugout_seed42.png`
* `tfg-notas/figuras/test_scores_drugout_seed42.json`

**TEST metrics**

* MSE: 0.04417116
* RMSE: 0.21016936
* PCC (Pearson): -0.02874616
* SCC (Spearman): -0.17842266
* R2: -0.60769871

**Conclusion:** performance drops sharply compared with the baseline. This indicates that the model generalizes **poorly** to unseen drugs. The model likely depends strongly on drug identity/representation, and there may also be a strong shift in the chemical space.

---

## 2026-03-13 — Ablation: Leave-Cell-Out (TEST) — `cellout_seed42`

**Objective:** evaluate generalization to **unseen cell lines** using held-out TEST cells.

In the output, the number of held-out TEST cells is reported as **60**.

**Experiment folder**

* `exp_result/runs/cellout_seed42/`

**Generated files (TEST)**

* `exp_result/runs/cellout_seed42/test_scores.json`
* `exp_result/runs/cellout_seed42/test_y_data_predicted.csv`
* `exp_result/runs/cellout_seed42/test_results.tsv`
* `exp_result/runs/cellout_seed42/test_scatter.png`

**Copies stored in notes**

* `tfg-notas/figuras/test_scatter_cellout_seed42.png`
* `tfg-notas/figuras/test_scores_cellout_seed42.json`

**TEST metrics**

* MSE: 0.00749011
* RMSE: 0.08654542
* PCC (Pearson): 0.83968955
* SCC (Spearman): 0.68857239
* R2: 0.70125897

**Conclusion:** performance decreases moderately compared with the baseline, but remains reasonable. This suggests that the model generalizes **better to unseen cell lines** than to unseen drugs.

**Stored files for `cellout_seed42`**

* `tfg-notas/figuras/test_scores_cellout_seed42.json`
* `tfg-notas/figuras/test_results_cellout_seed42.tsv`
* `tfg-notas/figuras/test_scatter_cellout_seed42.png`

## Global Conclusion

* In the normal split, based on new cell–drug pairs, the baseline performs best. The `heads4` and `layers4` ablations do not improve performance.
* In more demanding generalization scenarios, **leave-cell-out** maintains reasonable performance, while **leave-drug-out** drops drastically, with a negative R2. This indicates that generalizing to unseen drugs is the main challenge.

## Random Forest Baseline with ECFP4

A classical baseline was trained using:

* gene expression of the cell line;
* ECFP4 fingerprint of the drug.

The final model input contains 3006 variables:

* 958 gene expression variables;
* 2048 ECFP4 bits.

It was verified that the script correctly uses the matrix:

`gene_expression/block0_values`

and not the drug identifier matrix.

Test results for the normal split:

* MSE: 0.007130
* RMSE: 0.084412
* MAE: 0.064785
* PCC: 0.859304
* SCC: 0.749858
* R2: 0.738113

**Interpretation**

The Random Forest model with ECFP4 achieves performance close to the DeepTTC baseline on the normal split, although slightly lower. This indicates that a classical molecular representation such as ECFP4, combined with gene expression, already captures relevant information for predicting AUC in new cell–drug pairs.

**Generated files**

* `tfg-notas/experimentos_baselines/drug_ecfp4.tsv`
* `tfg-notas/experimentos_baselines/rf_ecfp_pair/test_scores.json`
* `tfg-notas/experimentos_baselines/rf_ecfp_pair/summary.json`
* `tfg-notas/experimentos_baselines/rf_ecfp_pair/test_predictions.tsv`

## Location of Latent Vectors in `Step3_model.py`

`Step3_model.py` was inspected to locate possible latent vectors in the model.

**Identified points**

* The drug branch returns `encoded_layers[:, 0]`, which can be interpreted as the latent representation of the drug.
* The concatenation `v_f = torch.cat((v_D, v_P), 1)` produces a joint cell–drug representation.
* The vector `v_f` is then passed through the final predictor to produce the AUC prediction.

**Conclusion**

It is possible to extract latent vectors from DeepTTC.

The most interesting vector for the analysis would be `fusion_latent = v_f`, because it internally represents the cell–drug pair just before the final prediction.

**Generated file**

* `tfg-notas/analisis_latentes/resumen_localizacion_latentes.md`

