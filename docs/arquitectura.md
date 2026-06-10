# Arquitectura del modelo (DeepTTC): rama fármaco + atención

## Visión general
DeepTTC predice la respuesta a fármacos (por defecto **AUC**) combinando **dos ramas**:
1) **Rama del fármaco (drug encoder)**: convierte el SMILES en un vector usando un **Transformer** con **self-attention multi-cabeza**.  
2) **Rama de la célula (gene encoder)**: procesa la **expresión génica** con un **MLP** y produce otro vector.  

Ambos vectores se **concatenan** y pasan por un **MLP final** que devuelve una única predicción (regresión).

**Esquema (alto nivel)**
- `SMILES → tokenización/ESPF → Transformer → z_drug`
- `genes → scaling → MLP → z_cell`
- `concat(z_drug, z_cell) → MLP → ŷ(AUC)`

---

## Rama del fármaco: SMILES → embeddings → Transformer (atención)
### 1) SMILES → representación de entrada
En el preprocesado, cada molécula está en formato **SMILES**. Se tokeniza (vía `Step2_DataEncoding.py`, método interno `obj._drug2emb_encoder`) y se guarda como `drug_encoding`.  
En el `forward` del modelo, el fármaco entra como:
- `e = v[0]`: **tokens** (IDs enteros)
- `e_mask = v[1]`: **máscara** (1 = token real, 0 = padding)

### 2) Embedding
Los tokens se proyectan a un espacio denso con:
- `input_dim_drug` = tamaño del vocabulario de tokens (p.ej. 2586)
- `transformer_emb_size_drug` = dimensión del embedding (p.ej. 128)

Esto convierte una secuencia de IDs en una secuencia de vectores.

### 3) Atención (Transformer encoder)
La parte “importante” del bloque de fármaco está en `Encoder_MultipleLayers(...)`, que implementa un Transformer con:
- `transformer_n_layer_drug`: nº de capas (p.ej. 8)
- `transformer_num_attention_heads_drug`: nº de cabezas de atención (p.ej. 8)
- `transformer_intermediate_size_drug`: tamaño de la red feed-forward interna (p.ej. 512)
- `transformer_attention_probs_dropout` y `transformer_hidden_dropout_rate`: dropouts

La máscara se expande y se convierte en el “mask” típico de Transformer (penalizando padding con valores muy negativos) para que la atención ignore tokens de relleno.

### 4) Vector final del fármaco
El Transformer devuelve una secuencia de embeddings (uno por token).  
El modelo usa el embedding de la **posición 0** como representación global de la molécula:

- `drug_embedding = encoded_layers[:, 0]`

En la práctica se usa como vector agregador (estilo **CLS**) aunque no sea un token especial explícito del SMILES.

---

## Rama de la célula: expresión génica → MLP
La expresión génica entra como un vector numérico (dimensión `gene_dim`). En nuestras ejecuciones el modelo reporta **958 genes de entrada** (subconjunto tipo LINCS).  
Se procesa con un MLP (`MLP`) con capas ocultas hasta obtener un embedding de célula (por defecto 256).

---

## Fusión y predicción final
En `Classifier.forward(...)`:
1) `v_D = model_drug(v_D)` → embedding del fármaco
2) `v_P = model_gene(v_P)` → embedding de la célula
3) `v_f = concat(v_D, v_P)` → concatenación
4) MLP final con capas ocultas (1024, 1024, 512) → salida escalar

La salida final es una predicción continua (regresión). Se entrena con pérdida **MSE** y se reportan métricas de test como **MSE/RMSE**, **Pearson/Spearman** y **R²** para comparar runs.

---

## Parámetros clave a reportar (bloque de fármacos)
- `input_dim_drug`
- `transformer_emb_size_drug`
- `transformer_n_layer_drug`
- `transformer_num_attention_heads_drug`
- `transformer_intermediate_size_drug`
- `transformer_attention_probs_dropout`
- `transformer_hidden_dropout_rate`
- `dropout`

---

## Ablation study (configuración del Transformer de fármacos)
Para analizar la sensibilidad del modelo a la configuración del Transformer que procesa los fármacos (SMILES → embeddings → atención), se realizaron dos ablations sobre el conjunto TEST:
- **heads4**: reducción del número de cabezas de atención (`transformer_num_attention_heads_drug`: 8 → 4).
- **layers4**: reducción del número de capas del encoder (`transformer_n_layer_drug`: 8 → 4).

Los resultados muestran que el **baseline (8 cabezas, 8 capas)** mantiene el mejor rendimiento global. Reducir cabezas a 4 apenas cambia las métricas (pero no mejora), mientras que reducir capas a 4 produce una degradación más clara (aumenta RMSE y disminuye R²). Esto sugiere que la **profundidad del encoder** aporta capacidad representacional relevante para capturar relaciones subestructura–respuesta.

**Material generado para documentación**
- Figuras (TEST): `tfg-notas/figuras/test_scatter_{baseline,heads4,layers4}.png`
- Tabla resumen (TEST): `tfg-notas/tabla_resultados.md`