### Results Table (TEST) — DeepTTC + IMPROVE

> Note: **Heads** and **Layers** refer to the *drug Transformer* in the SMILES branch.
> In **cellout_seed42** and **drugout_seed42**, the architecture is the same as the baseline configuration (8/8); what changes is the **type of data split**.

| Run            | Split                                    | Heads | Layers |        MSE |       RMSE |         PCC |         SCC |          R2 | Figure                                              |
| -------------- | ---------------------------------------- | ----: | -----: | ---------: | ---------: | ----------: | ----------: | ----------: | --------------------------------------------------- |
| baseline       | pair split (CCLE splits)                 |     8 |      8 | 0.00669110 | 0.08179915 |  0.86853303 |  0.76613500 |  0.75424541 | `tfg-notas/figuras/test_scatter_baseline.png`       |
| rf_ecfp_pair   | pair split (CCLE splits)                 |     - |      - | 0.00713033 | 0.08441126 |  0.85930449 |  0.74985771 |  0.73811324 | `tfg-notas/figuras/test_scatter_rf_ecfp_pair.png`   |
| heads4         | pair split (CCLE splits)                 |     4 |      8 | 0.00674747 | 0.08214296 |  0.86816928 |  0.76525122 |  0.75217517 | `tfg-notas/figuras/test_scatter_heads4.png`         |
| layers4        | pair split (CCLE splits)                 |     8 |      4 | 0.00695003 | 0.08336682 |  0.86561822 |  0.76372329 |  0.74473540 | `tfg-notas/figuras/test_scatter_layers4.png`        |
| cellout_seed42 | **leave-cell-out** (TEST cells held out) |     8 |      8 | 0.00749011 | 0.08654542 |  0.83968955 |  0.68857239 |  0.70125897 | `tfg-notas/figuras/test_scatter_cellout_seed42.png` |
| drugout_seed42 | **leave-drug-out** (TEST drugs held out) |     8 |      8 | 0.04417116 | 0.21016936 | -0.02874616 | -0.17842266 | -0.60769871 | `tfg-notas/figuras/test_scatter_drugout_seed42.png` |

**Note on `rf_ecfp_pair`:** this is a classical Random Forest baseline using **gene expression + ECFP4**. It does not use a Transformer architecture, which is why the Heads and Layers columns are shown as `-`.

The Random Forest baseline achieves performance close to DeepTTC on the normal pair split, although slightly lower. This indicates that a classical drug representation such as ECFP4 already captures part of the relevant information needed to predict AUC.
