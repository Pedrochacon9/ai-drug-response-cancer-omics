## Metodología reproducible (DeepTTC + IMPROVE)

1) **Entorno**: WSL Ubuntu con entorno conda `deepttc` y GPU disponible (`torch.cuda.is_available() = True`, `cuda:0`).

2) **Código**: repositorio principal `DeepTTC` (implementación del modelo) + `IMPROVE` instalado en editable (`pip install -e .`) para usar utilidades del estándar IMPROVE.

3) **Datos (IMPROVE/CSA)**: se utilizan los ficheros del benchmark:
- **Fármaco**: `csa_data/raw_data/x_data/drug_SMILES.tsv` (SMILES) y/o representaciones derivadas.
- **Ómica**: `csa_data/raw_data/x_data/cancer_gene_expression.tsv` (expresión génica).
- **Respuesta**: `csa_data/raw_data/y_data/response.tsv` (objetivo `auc`).

4) **Validación del preprocesamiento**  
Se verificó la calidad del conjunto preprocesado (train/val/test) comprobando:
- Ausencia de valores faltantes (NaNs = 0 en todas las particiones).
- Ausencia de duplicados del par (línea celular, fármaco).
- No solapamiento de pares entre splits (train∩val=0, train∩test=0, val∩test=0), evitando fuga de información (data leakage) a nivel de ejemplos.

**Tamaño del dataset y solapamientos (split normal por pares)**  
- Train: 7616 ejemplos, 411 líneas celulares, 24 fármacos.
- Val: 952 ejemplos, 371 líneas celulares, 24 fármacos.
- Test: 951 ejemplos, 371 líneas celulares, 24 fármacos.

Los pares (línea celular, fármaco) están **totalmente disjuntos** entre train/val/test.  
Sin embargo, existe solapamiento a nivel de entidades (los **24 fármacos** aparecen en los tres splits y hay solapamiento parcial de **líneas celulares**). Por tanto, este split evalúa generalización a **pares nuevos** más que a entidades completamente nuevas.

**Preprocesado**: se generan datasets en HDF5:
- `exp_result/train_data.h5`, `exp_result/val_data.h5`, `exp_result/test_data.h5`  
  (keys esperadas en el HDF5: `drug` y `gene_expression`).

5) **Entrenamiento**: se entrena el modelo y se guardan artefactos de la ejecución:
- Modelo: `model.pt`
- Logs: `train_*.log`
- (si aplica) scores en validación: `val_scores.json` / métricas impresas en log.

6) **Inferencia (TEST)**: se ejecuta inferencia sobre `test_data.h5` y se guardan predicciones y métricas:
- Predicciones: `test_y_data_predicted.csv`
- Métricas: `test_scores.json`
- Tabla (true vs pred): `test_results.tsv`
- Figura: `test_scatter.png` (generada en el run y copiada a `tfg-notas/figuras/` para documentación).

7) **Experimentos / ablations (runs)**: cada configuración se ejecuta en carpeta independiente:
- `exp_result/runs/<run>/` (p.ej. `baseline`, `heads4`, `layers4`, `drugout_seed42`, `cellout_seed42`)  
  manteniendo trazabilidad de `model.pt`, logs, `test_scores.json` y outputs de inferencia.

8) **Documentación**: se registra en bitácora y se genera material listo para memoria:
- `tfg-notas/bitacora.md`
- `tfg-notas/tabla_resultados.md`
- `tfg-notas/resultados_memoria.md`

**Reproducir un run (idea general)**  
Entrar en `exp_result/runs/<run>/` y ejecutar: (i) entrenamiento → (ii) inferencia TEST → (iii) copiar figura/scores a `tfg-notas/figuras/`.