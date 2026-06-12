### Tabla de resultados (TEST) — DeepTTC + IMPROVE

> Nota: **Heads** y **Layers** se refieren al *Transformer del fármaco* (rama SMILES).  
> En **cellout_seed42** y **drugout_seed42** la arquitectura es la misma que el baseline (8/8); lo que cambia es el **tipo de partición (split)**.

| Run | Split | Heads | Layers | MSE | RMSE | PCC | SCC | R2 | Figura |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | pair (splits CCLE) | 8 | 8 | 0.00669110 | 0.08179915 | 0.86853303 | 0.76613500 | 0.75424541 | `tfg-notas/figuras/test_scatter_baseline.png` |
| rf_ecfp_pair | pair (splits CCLE) | - | - | 0.00713033 | 0.08441126 | 0.85930449 | 0.74985771 | 0.73811324 | `tfg-notas/figuras/test_scatter_rf_ecfp_pair.png` |
| heads4 | pair (splits CCLE) | 4 | 8 | 0.00674747 | 0.08214296 | 0.86816928 | 0.76525122 | 0.75217517 | `tfg-notas/figuras/test_scatter_heads4.png` |
| layers4 | pair (splits CCLE) | 8 | 4 | 0.00695003 | 0.08336682 | 0.86561822 | 0.76372329 | 0.74473540 | `tfg-notas/figuras/test_scatter_layers4.png` |
| cellout_seed42 | **leave-cell-out** (TEST cells held-out) | 8 | 8 | 0.00749011 | 0.08654542 | 0.83968955 | 0.68857239 | 0.70125897 | `tfg-notas/figuras/test_scatter_cellout_seed42.png` |
| drugout_seed42 | **leave-drug-out** (TEST drugs held-out) | 8 | 8 | 0.04417116 | 0.21016936 | -0.02874616 | -0.17842266 | -0.60769871 | `tfg-notas/figuras/test_scatter_drugout_seed42.png` |

**Nota sobre rf_ecfp_pair:** es un baseline clásico Random Forest usando **expresión génica + ECFP4**. No usa arquitectura Transformer, por eso las columnas Heads y Layers aparecen como `-`.

El baseline Random Forest obtiene un rendimiento cercano a DeepTTC en el split normal, aunque ligeramente inferior. Esto indica que una representación clásica del fármaco como ECFP4 ya captura parte de la información relevante para predecir AUC.