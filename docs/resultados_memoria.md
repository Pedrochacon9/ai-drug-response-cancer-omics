## Resultados (DeepTTC + IMPROVE)

Se entrenó DeepTTC para predecir respuesta a fármacos usando el benchmark IMPROVE (expresión génica + representación del fármaco basada en SMILES). La evaluación se realizó sobre el conjunto TEST y se registraron métricas de regresión (MSE, RMSE, correlación de Pearson/Spearman y R²). Además, se generaron gráficas de dispersión AUC real vs AUC predicha para documentar visualmente la calidad de predicción.

En el estudio de ablation del bloque Transformer de fármaco, el **baseline (8 cabezas, 8 capas)** obtuvo el mejor rendimiento global. Reducir el número de cabezas a 4 (heads4) apenas altera los resultados pero no aporta mejora. En cambio, reducir el número de capas a 4 (layers4) degrada de forma más clara el rendimiento (sube RMSE y baja R²), lo que indica que la profundidad del encoder contribuye a capturar relaciones relevantes entre estructura del fármaco y respuesta.

### Limitaciones y trabajo futuro

En la partición actual, train/val/test comparten los mismos 24 fármacos y parte de las líneas celulares; por ello, la evaluación refleja principalmente generalización a **pares nuevos** (cell–drug). Como extensión, se plantea evaluar generalización más exigente mediante:
1) **Split por fármaco** (leave-drug-out): el modelo predice respuesta para fármacos no vistos en entrenamiento.
2) **Split por línea celular** (leave-cell-out): predicción en células no vistas.
Estas variantes permiten medir mejor la robustez del modelo en escenarios de descubrimiento/traslación.