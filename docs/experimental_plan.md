# Plan experimental del TFG

## Objetivo general

El objetivo del trabajo no es únicamente obtener una métrica final de predicción, sino analizar cómo influyen los datos de entrada, el preprocesamiento, la representación del fármaco y el tipo de partición en la predicción de respuesta farmacológica.

El problema estudiado consiste en predecir el valor de AUC para un par línea celular-fármaco usando:

- expresión génica de la línea celular;
- representación molecular del fármaco;
- respuesta observada AUC como variable objetivo.

## Bloques del trabajo

El trabajo se organiza en tres bloques:

1. Comprensión y preprocesamiento del dataset.
2. Comparación experimental de modelos.
3. Análisis e interpretación del comportamiento del modelo.

## Experimentos realizados

| Experimento | Modelo | Entrada | Split | Objetivo |
|---|---|---|---|---|
| E1 | DeepTTC | SMILES + expresión génica | pair split | Evaluar rendimiento en pares célula-fármaco nuevos |
| E2 | DeepTTC heads4 | SMILES + expresión génica | pair split | Ablación del número de cabezas de atención |
| E3 | DeepTTC layers4 | SMILES + expresión génica | pair split | Ablación del número de capas Transformer |
| E4 | DeepTTC | SMILES + expresión génica | cell-out | Evaluar generalización a líneas celulares no vistas |
| E5 | DeepTTC | SMILES + expresión génica | drug-out | Evaluar generalización a fármacos no vistos |

## Resultados observados hasta ahora

Los resultados muestran que el modelo funciona bien en el split normal de pares célula-fármaco y mantiene un rendimiento razonable en cell-out.  
Sin embargo, el rendimiento cae de forma importante en drug-out.

Esto sugiere que la generalización a fármacos completamente nuevos es el punto más difícil del problema.

## Experimentos propuestos

| Experimento | Modelo | Entrada | Split | Objetivo |
|---|---|---|---|---|
| E6 | Random Forest | expresión génica + ECFP4 | pair split | Baseline clásico |
| E7 | XGBoost | expresión génica + ECFP4 | pair split | Baseline clásico fuerte |
| E8 | Random Forest | expresión génica + ECFP4 | drug-out | Ver si el fallo en fármacos nuevos también aparece en un modelo clásico |
| E9 | XGBoost | expresión génica + ECFP4 | drug-out | Comparar generalización drug-out frente a DeepTTC |
| E10 | DeepTTC latentes | vectores internos | pair/drug-out | Analizar qué aprende internamente el modelo |

## Análisis complementarios

Además de las métricas globales, se realizarán análisis complementarios:

- distribución de AUC;
- número de ejemplos por fármaco;
- número de ejemplos por línea celular;
- solape de entidades entre splits;
- error por fármaco en drug-out;
- comparación entre AUC real y AUC predicho;
- extracción y visualización de vectores latentes.

## Justificación

El split normal evalúa pares célula-fármaco nuevos, pero no evalúa fármacos completamente nuevos porque las mismas drogas aparecen en train, validación y test.  
Por eso es necesario estudiar splits más exigentes, especialmente drug-out.

El análisis drug-out permite evaluar si el modelo aprende una representación generalizable del fármaco o si depende demasiado de haber visto previamente las drogas durante el entrenamiento.


## Resultado preliminar de Random Forest con ECFP4

Se ha ejecutado un primer baseline clásico usando Random Forest con:

`expresión génica + ECFP4`

La entrada final tiene 3006 características:

- 958 genes;
- 2048 bits ECFP4.

Resultado en test sobre el split normal:

| Modelo | Entrada | Split | RMSE | PCC | R2 |
|---|---|---|---:|---:|---:|
| DeepTTC baseline | SMILES + genes | pair split | 0.0818 | 0.8685 | 0.7542 |
| Random Forest | ECFP4 + genes | pair split | 0.0844 | 0.8593 | 0.7381 |

Este resultado muestra que el baseline clásico queda cerca de DeepTTC en el split normal, aunque ligeramente por debajo.

