## Descripción del modelo (DeepTTC)

DeepTTC es un modelo de predicción de respuesta a fármacos que combina información del **fármaco** (en formato SMILES) y de la **línea celular** (expresión génica) para predecir una variable continua de respuesta (en nuestro caso, **AUC**). La idea central es aprender una representación numérica del fármaco a partir de SMILES mediante un bloque tipo **Transformer**, y aprender una representación compacta de la célula a partir del vector de genes mediante una **red MLP**.

### Rama del fármaco (SMILES → Transformer)
El SMILES se convierte primero en una secuencia de tokens (subestructuras) y se proyecta a embeddings. Esa secuencia se procesa con un **encoder Transformer** (multi-head self-attention + capas feed-forward), produciendo un vector resumen del fármaco. En nuestras ejecuciones, los hiperparámetros relevantes de esta rama son el número de cabezas de atención (`transformer_num_attention_heads_drug`) y el número de capas del encoder (`transformer_n_layer_drug`), que se analizaron mediante ablations.

### Rama de la célula (expresión génica → MLP)
La expresión génica se representa como un vector de dimensión fija (en nuestro caso se utilizan **958 genes**). Ese vector se introduce en una **MLP** con varias capas densas y activación ReLU para obtener una representación latente de la línea celular, que captura patrones de sensibilidad/resistencia asociados al perfil transcriptómico.

### Fusión y salida (regresión)
Los vectores latentes de fármaco y célula se **concatenan** y se pasan por un MLP final (“classifier” en el código, aunque aquí realiza regresión) que produce una única salida real: la predicción de **AUC**. El entrenamiento se realiza minimizando **MSE** y se evalúa en TEST con métricas de regresión (MSE/RMSE), correlaciones (Pearson/Spearman) y **R²**.