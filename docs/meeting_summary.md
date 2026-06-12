# Resumen para enseñar en la reunión

## 1. Reorganización del enfoque

El trabajo ya no se plantea solo como una tabla de resultados, sino como un análisis completo del pipeline:

1. entender y preprocesar los datos;
2. comparar modelos;
3. analizar errores e interpretar el comportamiento del modelo.

## 2. Notebook explicativo del dataset

Se ha creado:

notebooks_tfg/01_entender_dataset.ipynb

Este notebook explica:

- de dónde vienen los datos;
- qué es una línea celular;
- qué es un fármaco;
- qué es un SMILES;
- qué es la expresión génica;
- qué es el AUC;
- cómo se forma una muestra de entrenamiento.

También permite ver que el fichero raw de expresión génica contiene filas iniciales de metadatos, por lo que el preprocesamiento es importante.

## 3. Análisis drug-out

Se ha analizado el experimento drug-out más allá de la métrica global.

Archivos principales:

- tfg-notas/analisis_drugout/resumen_interpretativo_drugout.md
- tfg-notas/figuras/drugout_rmse_por_farmaco.png
- tfg-notas/figuras/drugout_scatter_real_predicho.png
- tfg-notas/figuras/drugout_error_absoluto_hist.png

Conclusión principal:

El error no está repartido igual entre todos los fármacos. Drug_490 concentra mucho error. Además, en el scatter aparecen bandas horizontales, lo que sugiere que en drug-out el modelo tiende a predecir ciertos niveles de AUC y no captura bien la variabilidad real de algunos fármacos no vistos.

## 4. ECFP4 y baseline clásico

Se ha generado una representación alternativa del fármaco:

tfg-notas/experimentos_baselines/drug_ecfp4.tsv

Resultado:

- 1565 fármacos convertidos.
- 2048 bits ECFP4 por fármaco.
- 0 SMILES inválidos.

También se ha entrenado un Random Forest usando:

expresión génica + ECFP4

Archivos principales:

- tfg-notas/experimentos_baselines/rf_ecfp_pair/test_scores.json
- tfg-notas/experimentos_baselines/rf_ecfp_pair/summary.json

Resultado en test:

- RMSE: 0.0844
- PCC: 0.8593
- R2: 0.7381

Comparación:

DeepTTC baseline tiene RMSE 0.0818, por lo que Random Forest queda cerca, aunque ligeramente peor.

## 5. Vectores latentes

Se ha inspeccionado Step3_model.py y se han localizado puntos claros para extraer vectores latentes.

Archivos principales:

- tfg-notas/analisis_latentes/localizacion_latentes.txt
- tfg-notas/analisis_latentes/resumen_localizacion_latentes.md

Puntos localizados:

1. encoded_layers[:, 0] puede interpretarse como vector latente del fármaco.
2. v_f = torch.cat((v_D, v_P), 1) puede interpretarse como vector latente fusionado célula-fármaco.

El vector más interesante sería fusion_latent = v_f, porque representa el par célula-fármaco justo antes de la predicción final de AUC.

## 6. Siguientes pasos

Los siguientes pasos naturales serían:

1. entrenar Random Forest o XGBoost también en drug-out;
2. implementar de forma controlada return_latent=True en DeepTTC;
3. extraer fusion_latent y analizarlo con PCA/t-SNE;
4. estudiar si fármacos problemáticos como Drug_490 aparecen separados en el espacio latente;
5. repetir algunos experimentos con otras seeds para comprobar estabilidad.

