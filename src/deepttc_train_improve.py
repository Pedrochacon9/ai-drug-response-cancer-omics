import sys
from pathlib import Path
from typing import Dict

# Librerías generales del sistema y tratamiento de datos.
# Se utilizan para gestionar rutas, leer ficheros HDF5 y trabajar con parámetros.
import os
import json
import pandas as pd

# Importación de la arquitectura principal del modelo DeepTTC.
# En Step3_model.py se define la clase DeepTTC y sus métodos de entrenamiento,
# guardado, carga e inferencia.
from Step3_model import *

# Importaciones de IMPROVE necesarias para inicializar la configuración,
# generar nombres estándar de ficheros, guardar predicciones y calcular métricas.
from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
from improvelib.metrics import compute_metrics

# Parámetros adicionales específicos de entrenamiento definidos para este modelo.
from model_params_def import train_params


# Ruta base del script. IMPROVE la utiliza para localizar los ficheros
# de configuración del modelo y recursos asociados.
filepath = Path(__file__).resolve().parent


def get_model(args, gene_dim=958):
    """
    Construye una instancia del modelo DeepTTC.

    El número de variables de expresión génica puede variar según el
    preprocesamiento realizado. Por ello, la dimensión de entrada de la rama
    génica se obtiene dinámicamente a partir de los datos procesados.

    Args:
        args (dict): Parámetros de configuración del modelo.
        gene_dim (int): Número de variables de expresión génica de entrada.

    Returns:
        DeepTTC: Instancia del modelo DeepTTC preparada para entrenamiento.
    """
    net = DeepTTC(
        modeldir=args['output_dir'],
        args=args,
        gene_dim=gene_dim
    )
    return net


def run(params: Dict):
    """
    Ejecuta el entrenamiento del modelo DeepTTC.

    El flujo general seguido por esta función es:
    1. Construir los nombres de los ficheros de train y validación.
    2. Cargar las matrices de entrada ya preprocesadas.
    3. Construir el modelo DeepTTC con la dimensión génica correcta.
    4. Entrenar el modelo usando train y validación.
    5. Guardar el modelo entrenado.
    6. Cargar el modelo guardado y calcular predicciones en validación.
    7. Guardar predicciones y métricas de rendimiento.

    Args:
        params (dict): Diccionario de parámetros de IMPROVE y del modelo.

    Returns:
        dict: Métricas de rendimiento calculadas sobre el conjunto de validación.
    """

    # --------------------------------------------------------------------
    # Construcción de nombres de ficheros y ruta del modelo
    # --------------------------------------------------------------------
    # IMPROVE utiliza nombres estándar para los ficheros generados durante
    # el preprocesamiento. Aquí se obtiene el nombre esperado para train y val.
    train_data_fname = frm.build_ml_data_file_name(
        data_format=params["data_format"],
        stage="train"
    )

    val_data_fname = frm.build_ml_data_file_name(
        data_format=params["data_format"],
        stage="val"
    )

    # Ruta final donde se guardará el modelo entrenado.
    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["output_dir"]
    )

    # --------------------------------------------------------------------
    # Carga de datos preprocesados
    # --------------------------------------------------------------------
    # Los datos se guardaron previamente en formato HDF5 durante el
    # preprocesamiento. Cada fichero contiene al menos dos bloques:
    # - 'drug': información/codificación del fármaco;
    # - 'gene_expression': expresión génica de la línea celular.
    train_data = {}
    train_data['drug'] = pd.read_hdf(
        os.path.join(params["input_dir"], train_data_fname),
        key='drug'
    )

    train_data['gene_expression'] = pd.read_hdf(
        os.path.join(params["input_dir"], train_data_fname),
        key='gene_expression'
    )

    val_data = {}
    val_data['drug'] = pd.read_hdf(
        os.path.join(params["input_dir"], val_data_fname),
        key='drug'
    )

    val_data['gene_expression'] = pd.read_hdf(
        os.path.join(params["input_dir"], val_data_fname),
        key='gene_expression'
    )

    # --------------------------------------------------------------------
    # Preparación del modelo
    # --------------------------------------------------------------------
    # La dimensión de entrada de la rama génica se obtiene directamente
    # del fichero de entrenamiento. En este TFG normalmente son 958 genes,
    # pero se calcula automáticamente para evitar errores si cambia el dataset.
    input_gene_dim = train_data['gene_expression'].shape[1]
    print(f'Number of genes of input gene expression data: {input_gene_dim}')

    model = get_model(params, gene_dim=input_gene_dim)

    # --------------------------------------------------------------------
    # Entrenamiento
    # --------------------------------------------------------------------
    # Se entrena DeepTTC usando las dos entradas principales:
    # - train_drug: representación/codificación del fármaco;
    # - train_rna: expresión génica de la línea celular.
    #
    # Además, se pasa el conjunto de validación para monitorizar el rendimiento
    # durante el entrenamiento.
    model = model.train(
        train_drug=train_data['drug'],
        train_rna=train_data['gene_expression'],
        val_drug=val_data['drug'],
        val_rna=val_data['gene_expression']
    )

    # --------------------------------------------------------------------
    # Guardado del modelo
    # --------------------------------------------------------------------
    # Una vez finalizado el entrenamiento, se guarda el modelo en la ruta
    # definida por los parámetros de IMPROVE/DeepTTC.
    print(f'Saving model to {modelpath}')
    model.save_model(modelpath)
    print("Model Saved :{}".format(modelpath))

    # --------------------------------------------------------------------
    # Carga del modelo guardado y predicción sobre validación
    # --------------------------------------------------------------------
    # Se carga el modelo guardado para asegurar que el fichero generado
    # puede reutilizarse posteriormente en inferencia.
    model.load_pretrained(modelpath)

    # La función predict devuelve etiquetas reales, predicciones y varias
    # métricas internas calculadas por el propio modelo.
    y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI = model.predict(
        val_data['drug'],
        val_data['gene_expression']
    )

    # --------------------------------------------------------------------
    # Guardado de predicciones
    # --------------------------------------------------------------------
    # Se almacenan las predicciones crudas de validación en un DataFrame.
    # Esto permite analizarlas posteriormente o compararlas con otros runs.
    frm.store_predictions_df(
        y_true=y_label,
        y_pred=y_pred,
        stage="val",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_dir"]
    )

    # --------------------------------------------------------------------
    # Cálculo y guardado de métricas
    # --------------------------------------------------------------------
    # Se calculan las métricas de rendimiento de validación usando el formato
    # estándar de IMPROVE. Estas métricas permiten comparar runs de forma homogénea.
    val_scores = frm.compute_performance_scores(
        y_true=y_label,
        y_pred=y_pred,
        stage="val",
        output_dir=params["output_dir"],
        metric_type=params["metric_type"]
    )

    return val_scores


def main(args):
    """
    Punto de entrada del script de entrenamiento.

    Inicializa la configuración de entrenamiento mediante IMPROVE, carga los
    parámetros definidos para DeepTTC y ejecuta el entrenamiento completo.
    """

    filepath = Path(__file__).resolve().parent

    # Configuración estándar de entrenamiento para Drug Response Prediction.
    cfg = DRPTrainConfig()

    # Inicialización de parámetros:
    # - deepttc_params.txt contiene la configuración general del modelo;
    # - train_params añade o modifica parámetros específicos del entrenamiento.
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="deepttc_params.txt",
        additional_definitions=train_params
    )

    # Ejecución del entrenamiento y obtención de métricas sobre validación.
    val_scores = run(params)

    print("\nFinished training model.")


if __name__ == "__main__":
    main(sys.argv[1:])