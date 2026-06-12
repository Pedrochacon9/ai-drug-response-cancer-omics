# Dataset / Memoria de datos — DeepTTC + IMPROVE

Estamos usando el **benchmark de IMPROVE (drug response prediction)** como fuente estándar de datos, y el repo **DeepTTC** lo consume en formato “IMPROVE-style”: parte de los datos vienen ya en `csa_data/raw_data/` (features X e Y), y otra parte son recursos `author_data/` (vocabulario/ESPF) necesarios para codificar SMILES. En nuestro flujo, primero **preprocesamos** (generamos `train_data.h5`, `val_data.h5`, `test_data.h5` y CSVs auxiliares) y luego entrenamos/inferimos usando esos ficheros ya preparados.

## Repos / rutas principales (WSL)
- Repo modelo: `~/tfg/DeepTTC/`
- Repo IMPROVE (librería): `~/tfg/IMPROVE/` *(instalado con `pip install -e .` para poder hacer `import improvelib`)*

## Ficheros de datos “crudos” que realmente usamos (DeepTTC)
### Target (Y)
- `csa_data/raw_data/y_data/response.tsv`  
  Contiene la respuesta/label por par **(improve_sample_id, improve_chem_id)**.  
  En nuestros runs el target es `auc`.

### Drug (X-drug)
- `csa_data/raw_data/x_data/drug_SMILES.tsv`  
  Mapa **improve_chem_id → SMILES/canSMILES** (entrada del transformer de fármacos).
- (alternativos disponibles, no imprescindibles si usamos SMILES):
  - `csa_data/raw_data/x_data/drug_ecfp4_nbits512.tsv`
  - `csa_data/raw_data/x_data/drug_mordred.tsv`
  - `csa_data/raw_data/x_data/drug_info.tsv`

### Cell line / cáncer (X-cell)
- `csa_data/raw_data/x_data/cancer_gene_expression.tsv`  
  Matriz de expresión génica por `improve_sample_id` (en nuestro caso, el modelo reporta **958 genes** de entrada).

### Splits (train/val/test)
- `CCLE_split_0_train.txt`
- `CCLE_split_0_val.txt`
- `CCLE_split_0_test.txt`

## Recursos necesarios para codificar SMILES (author_data)
- `author_data/ESPF/`  
  Vocabulario y mapas de subwords (ESPF) usados para tokenizar SMILES.
  - `author_data/ESPF/subword_units_map_uniprot_2000.csv`
  - `author_data/ESPF/subword_units_map_chembl_freq_1500.csv`

## Salidas del preprocesamiento (las que se usan en entrenamiento/inferencia)
Se guardan en `exp_result/` (o dentro de `exp_result/runs/<run>/` si hacemos runs separados):
- `train_data.h5`, `val_data.h5`, `test_data.h5`
- `train_y_data.csv`, `val_y_data.csv`, `test_y_data.csv`

## Salidas típicas de entrenamiento/inferencia
- Modelo: `model.pt`
- Scores: `val_scores.json`, `test_scores.json`
- Predicciones: `val_y_data_predicted.csv`, `test_y_data_predicted.csv`
- Tabla (true vs pred): `val_results.tsv`, `test_results.tsv`
- Figura: `val_scatter.png`, `test_scatter.png`

---

## Dataset elegido (para explicarlo al profesor)

Estamos usando el **benchmark IMPROVE (Drug Response Prediction)**. El origen del benchmark es **IMPROVE/CANDLE (Argonne National Lab)**; en nuestro proyecto estos ficheros aparecen ya organizados bajo `csa_data/raw_data/` y el pipeline los preprocesa a HDF5 para entrenar/inferir. Los datos “crudos” que consume DeepTTC (SMILES de fármacos, expresión génica de líneas celulares y la respuesta AUC por par cell–drug) están en `csa_data/raw_data/`, y DeepTTC está preparado para trabajar con ese formato estándar “IMPROVE-style”. Además, para ejecutar el flujo tal y como lo espera el framework, instalamos la **librería `improvelib`** desde el repo **IMPROVE** (`~/tfg/IMPROVE/` con `pip install -e .`), que aporta utilidades y estructura para entrenamiento/inferencia bajo el estándar IMPROVE.

- **Dataset (datos):** los ficheros de `csa_data/raw_data/` (X: `drug_SMILES.tsv`, `cancer_gene_expression.tsv`; Y: `response.tsv`) + los splits `CCLE_split_0_*.txt`.
- **Recursos de codificación (no son datos del benchmark, pero son necesarios):** `author_data/ESPF/` (vocabulario ESPF para tokenizar SMILES).
- **Tooling/estándar (código, no dataset):** repo `IMPROVE/` (instalado como `improvelib`) que permite ejecutar el flujo estilo IMPROVE.

### Cómo se obtiene / reproduce
- Repo DeepTTC: clonado en `~/tfg/DeepTTC/` (incluye `csa_data/raw_data/`).
- Repo IMPROVE: clonado en `~/tfg/IMPROVE/` e instalado en editable: `pip install -e .` (para que funcione `import improvelib`).
- Preprocesamiento genera los `.h5` y CSVs en `exp_result/` (o `exp_result/runs/<run>/`).

### Nota sobre particiones (splits)
- Split “normal” (pares nuevos): `exp_result/` (train/val/test sin solape de pares cell–drug).
- Splits de generalización: `exp_result/runs/drugout_seed42/` (leave-drug-out) y `exp_res