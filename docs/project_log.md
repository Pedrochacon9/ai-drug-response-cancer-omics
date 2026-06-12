## 2026-03-13 — Inferencia en TEST (DeepTTC + IMPROVE)

**Entorno**
- WSL Ubuntu + conda env: `deepttc`
- GPU: `torch.cuda.is_available() = True` (cuda:0)

**Entradas usadas**
- Datos procesados: `exp_result/test_data.h5`
- Modelo: `exp_result/model.pt`

**Comando de inferencia (TEST)**
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

**Archivos generados**
- `exp_result/test_y_data_predicted.csv`
- `exp_result/test_scores.json`
- `exp_result/test_results.tsv`
- `exp_result/test_scatter.png`

**Métricas (TEST)**
- MSE: 0.00962566596353434
- RMSE: 0.09811047835748402
- PCC (Pearson): 0.8140919644360205
- SCC (Spearman): 0.6676354143741668
- R2: 0.6464630026737932

---

## 2026-03-13 — Ablation: nº de cabezas de atención (TEST)

**Objetivo:** comprobar impacto de reducir `transformer_num_attention_heads_drug` de 8 → 4.

**Resultados en TEST (test_scores.json):**

| Experimento | MSE | RMSE | PCC | SCC | R2 |
|---|---:|---:|---:|---:|---:|
| baseline (heads=8) | 0.00669110 | 0.08179915 | 0.86853303 | 0.76613500 | 0.75424541 |
| heads4 (heads=4)   | 0.00674747 | 0.08214296 | 0.86816928 | 0.76525122 | 0.75217517 |

**Carpetas de ejecución (runs)**
- `exp_result/runs/baseline/`
- `exp_result/runs/heads4/`

**Conclusión:** reducir cabezas 8→4 no mejora (ligeramente peor en MSE/RMSE/R2).

---

## 2026-03-13 — Ablation: nº de capas del Transformer de fármaco (TEST)

**Objetivo:** comprobar impacto de reducir `transformer_n_layer_drug` de 8 → 4.

**Resultado (layers4, TEST)**
- MSE: 0.00695003
- RMSE: 0.08336682
- PCC (Pearson): 0.86561822
- SCC (Spearman): 0.76372329
- R2: 0.74473540

**Carpeta de ejecución (run)**
- `exp_result/runs/layers4/`

**Conclusión:** bajar capas 8→4 empeora ligeramente respecto al baseline (peor MSE/RMSE/PCC/SCC/R2).

---

## 2026-03-13 — Comparativa final de ablations (TEST)

**Runs comparados**
- `exp_result/runs/baseline/`  (heads=8, layers=8)
- `exp_result/runs/heads4/`    (heads=4, layers=8)
- `exp_result/runs/layers4/`   (heads=8, layers=4)

**Métricas (TEST)**

| Run | Heads | Layers | MSE | RMSE | PCC | SCC | R2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 8 | 8 | 0.00669110 | 0.08179915 | 0.86853303 | 0.76613500 | 0.75424541 |
| heads4   | 4 | 8 | 0.00674747 | 0.08214296 | 0.86816928 | 0.76525122 | 0.75217517 |
| layers4  | 8 | 4 | 0.00695003 | 0.08336682 | 0.86561822 | 0.76372329 | 0.74473540 |

**Conclusión**
- El **baseline** (8 cabezas, 8 capas) rinde mejor.
- Reducir cabezas a 4 apenas cambia, pero **no mejora**.
- Reducir capas a 4 **empeora más** (sube RMSE y baja R2).

**Figuras para memoria**
- `tfg-notas/figuras/test_scatter_baseline.png`
- `tfg-notas/figuras/test_scatter_heads4.png`
- `tfg-notas/figuras/test_scatter_layers4.png`

**Tabla resumen para memoria**
- `tfg-notas/tabla_resultados.md`

**Resumen para memoria**
- Tabla: `tfg-notas/tabla_resultados.md`
- Texto: `tfg-notas/resultados_memoria.md`
- Figuras: `tfg-notas/figuras/`

**Metodología reproducible**
- `tfg-notas/metodologia_reproducible.md`

**Modelo (texto para memoria)**
- `tfg-notas/modelo_memoria.md`


---

## 2026-03-16 — Ablation: leave-drug-out (TEST) — `drugout_seed42`

**Objetivo:** evaluar generalización a **fármacos no vistos** (se retiran fármacos del entrenamiento; el TEST contiene esos fármacos held-out).

**Carpeta del experimento**
- `exp_result/runs/drugout_seed42/`
- Lista de fármacos held-out: `exp_result/runs/drugout_seed42/heldout_drugs.txt`

**Archivos generados (TEST)**
- `exp_result/runs/drugout_seed42/test_scores.json`
- `exp_result/runs/drugout_seed42/test_y_data_predicted.csv`
- `exp_result/runs/drugout_seed42/test_results.tsv`
- `exp_result/runs/drugout_seed42/test_scatter.png`
- Copias en notas:
  - `tfg-notas/figuras/test_scatter_drugout_seed42.png`
  - `tfg-notas/figuras/test_scores_drugout_seed42.json`

**Métricas (TEST)**
- MSE: 0.04417116
- RMSE: 0.21016936
- PCC (Pearson): -0.02874616
- SCC (Spearman): -0.17842266
- R2: -0.60769871

**Conclusión:** caída muy fuerte respecto a baseline ⇒ generaliza **mal** a fármacos no vistos (el modelo depende bastante de “identidad/representación del fármaco” y/o hay shift fuerte en el espacio químico).

---

## 2026-03-13 — Ablation: leave-cell-out (TEST) — `cellout_seed42`

**Objetivo:** evaluar generalización a **líneas celulares no vistas** (held-out TEST cells).  
En tu salida aparece: **Held-out TEST cells = 60**.

**Carpeta del experimento**
- `exp_result/runs/cellout_seed42/`

**Archivos generados (TEST)**
- `exp_result/runs/cellout_seed42/test_scores.json`
- `exp_result/runs/cellout_seed42/test_y_data_predicted.csv`
- `exp_result/runs/cellout_seed42/test_results.tsv`
- `exp_result/runs/cellout_seed42/test_scatter.png`
- Copias en notas:
  - `tfg-notas/figuras/test_scatter_cellout_seed42.png`
  - `tfg-notas/figuras/test_scores_cellout_seed42.json`

**Métricas (TEST)**
- MSE: 0.00749011
- RMSE: 0.08654542
- PCC (Pearson): 0.83968955
- SCC (Spearman): 0.68857239
- R2: 0.70125897

**Conclusión:** baja moderada vs baseline (pero sigue “bien”) ⇒ el modelo generaliza **mejor a células nuevas** que a fármacos nuevos.

**Archivos guardados (cellout_seed42)**
- `tfg-notas/figuras/test_scores_cellout_seed42.json`
- `tfg-notas/figuras/test_results_cellout_seed42.tsv`
- `tfg-notas/figuras/test_scatter_cellout_seed42.png

## Conclusión global (resumen)

- En el split “normal” (pares nuevos), el baseline rinde mejor; las ablations **heads4** y **layers4** no mejoran.
- En escenarios de generalización más exigentes, **leave-cell-out** mantiene un rendimiento razonable, mientras que **leave-drug-out** cae drásticamente (R² negativo), indicando que generalizar a fármacos no vistos es el principal reto.
## Baseline Random Forest con ECFP4

Se ha entrenado un baseline clásico usando:

- expresión génica de la línea celular;
- fingerprint ECFP4 del fármaco.

La entrada final del modelo tiene 3006 variables:

- 958 variables de expresión génica;
- 2048 bits ECFP4.

Se verificó que el script usa correctamente la matriz:

`gene_expression/block0_values`

y no la matriz de identificadores de droga.

Resultados en test para el split normal:

- MSE: 0.007130
- RMSE: 0.084412
- MAE: 0.064785
- PCC: 0.859304
- SCC: 0.749858
- R2: 0.738113

Interpretación:

El Random Forest con ECFP4 obtiene un rendimiento cercano al baseline de DeepTTC en el split normal, aunque ligeramente inferior. Esto indica que una representación molecular clásica como ECFP4 combinada con expresión génica ya captura información relevante para predecir AUC en pares célula-fármaco nuevos.

Archivos generados:

- `tfg-notas/experimentos_baselines/drug_ecfp4.tsv`
- `tfg-notas/experimentos_baselines/rf_ecfp_pair/test_scores.json`
- `tfg-notas/experimentos_baselines/rf_ecfp_pair/summary.json`
- `tfg-notas/experimentos_baselines/rf_ecfp_pair/test_predictions.tsv`


## Localización de vectores latentes en Step3_model.py

Se ha inspeccionado `Step3_model.py` para localizar posibles vectores latentes del modelo.

Puntos localizados:

- La rama del fármaco devuelve `encoded_layers[:, 0]`, que puede interpretarse como representación latente del fármaco.
- La concatenación `v_f = torch.cat((v_D, v_P), 1)` genera una representación conjunta célula-fármaco.
- El vector `v_f` pasa después por el predictor final para producir la predicción AUC.

Conclusión:

Sí es posible extraer vectores latentes de DeepTTC.  
El vector más interesante para el análisis sería `fusion_latent = v_f`, porque representa internamente el par célula-fármaco justo antes de la predicción final.

Archivo generado:

- `tfg-notas/analisis_latentes/resumen_localizacion_latentes.md`

