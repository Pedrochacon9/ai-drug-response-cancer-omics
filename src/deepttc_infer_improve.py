import sys
from pathlib import Path
from typing import Dict

# Importaciones principales de IMPROVE para inferencia en tareas de
# Drug Response Prediction. DRPInferConfig inicializa la configuración
# necesaria para cargar datos, modelo y parámetros de inferencia.
from improvelib.applications.drug_response_prediction.config import DRPInferConfig
from improvelib.utils import str2bool
import improvelib.utils as frm

# Parámetros adicionales específicos para la fase de inferencia.
from model_params_def import infer_params

# Importación de la implementación del modelo DeepTTC.
# En este script se utiliza para cargar el modelo previamente entrenado
# y generar predicciones sobre el conjunto de test.
from DeepTTC_candle import *

# Librerías generales para rutas, uso de GPU/CPU y carga de datos.
import os
import torch
import pandas as pd


# Ruta base del script. IMPROVE la utiliza como referencia para localizar
# ficheros de configuración y recursos asociados al modelo.
filepath = Path(__file__).resolve().parent


def determine_device(cuda_name_from_params):
    """
    Determina si la inferencia se ejecutará en GPU o CPU.

    Esta función comprueba si PyTorch detecta CUDA disponible. Si existe una
    variable de entorno CUDA_VISIBLE_DEVICES, se respeta dicha configuración.
    En caso contrario, se usa el dispositivo indicado en los parámetros.

    En el flujo final del TFG, la selección de dispositivo puede estar gestionada
    directamente por los parámetros de DeepTTC, pero se mantiene esta función
    como utilidad para controlar la ejecución en GPU/CPU.

    Args:
        cuda_name_from_params (str): Nombre del dispositivo CUDA indicado
                                     en los parámetros, por ejemplo 'cuda:0'.

    Returns:
        str: Dispositivo seleccionado: 'cuda:0', otro identificador CUDA o 'cpu'.
    """
    cuda_avail = torch.cuda.is_available()
    print("GPU available: ", cuda_avail)

    if cuda_avail:
        cuda_env_visible = os.getenv("CUDA_VISIBLE_DEVICES")

        if cuda_env_visible is not None:
            # Si CUDA_VISIBLE_DEVICES está definido, PyTorch reindexa los
            # dispositivos visibles empezando desde cuda:0.
            print("CUDA_VISIBLE_DEVICES: ", cuda_env_visible)
            cuda_name = "cuda:0"
        else:
            cuda_name = cuda_name_from_params

        device = cuda_name

    else:
        device = "cpu"

    return device


def run(params: Dict):
    """
    Ejecuta la inferencia del modelo DeepTTC sobre el conjunto de test.

    El flujo seguido por esta función es:
    1. Construir el nombre del fichero de test preprocesado.
    2. Construir la ruta del modelo entrenado.
    3. Cargar las entradas de test: fármaco y expresión génica.
    4. Crear el modelo DeepTTC con la dimensión génica correcta.
    5. Cargar los pesos del modelo entrenado.
    6. Generar predicciones sobre test.
    7. Guardar predicciones y, si procede, calcular métricas finales.

    Args:
        params (dict): Diccionario de parámetros de IMPROVE y DeepTTC.

    Returns:
        bool: True si la inferencia termina correctamente.
    """

    # --------------------------------------------------------------------
    # Construcción del nombre del fichero de test y ruta del modelo
    # --------------------------------------------------------------------
    # IMPROVE genera nombres estándar para los ficheros de datos procesados.
    # Aquí se obtiene el nombre correspondiente al conjunto de test.
    test_data_fname = frm.build_ml_data_file_name(
        params['data_format'],
        stage="test"
    )

    # Ruta completa del modelo previamente entrenado.
    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["input_model_dir"]
    )

    # --------------------------------------------------------------------
    # Carga de datos de inferencia
    # --------------------------------------------------------------------
    # El fichero HDF5 de test contiene las dos entradas principales del modelo:
    # - 'drug': representación/codificación del fármaco;
    # - 'gene_expression': expresión génica de la línea celular.
    test_data = {}

    test_data['drug'] = pd.read_hdf(
        os.path.join(params["input_data_dir"], test_data_fname),
        key='drug'
    )

    test_data['gene_expression'] = pd.read_hdf(
        os.path.join(params["input_data_dir"], test_data_fname),
        key='gene_expression'
    )

    # --------------------------------------------------------------------
    # Selección de dispositivo
    # --------------------------------------------------------------------
    # La llamada se deja comentada porque en este flujo el uso de CUDA/CPU
    # se gestiona desde los parámetros y la propia implementación del modelo.
    # device = determine_device(params["cuda_name"])

    # --------------------------------------------------------------------
    # Carga del modelo y cálculo de predicciones
    # --------------------------------------------------------------------
    # Se obtiene dinámicamente el número de genes de entrada.
    # Esto evita fijar manualmente una dimensión que podría cambiar si se
    # modifica el preprocesamiento o el subconjunto de genes utilizado.
    input_gene_dim = test_data['gene_expression'].shape[1]
    print(f'Number of genes of input gene expression data: {input_gene_dim}')

    # Se construye el modelo con la misma dimensión de entrada que los datos.
    model = DeepTTC(
        modeldir=modelpath,
        args=params,
        gene_dim=input_gene_dim
    )

    # Carga de los pesos entrenados.
    model.load_pretrained(modelpath)

    # Generación de predicciones sobre el conjunto de test.
    # La función devuelve etiquetas reales, predicciones y métricas internas.
    y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI = model.predict(
        test_data['drug'],
        test_data['gene_expression']
    )

    # --------------------------------------------------------------------
    # Guardado de predicciones
    # --------------------------------------------------------------------
    # Se almacenan las predicciones crudas en formato DataFrame para poder
    # analizarlas posteriormente y generar figuras como los scatter plots.
    frm.store_predictions_df(
        y_true=y_label,
        y_pred=y_pred,
        stage="test",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_data_dir"]
    )

    # --------------------------------------------------------------------
    # Cálculo de métricas finales
    # --------------------------------------------------------------------
    # Si calc_infer_scores está activado, se calculan métricas de rendimiento
    # sobre test: MSE, RMSE, correlaciones y otras métricas configuradas.
    if params["calc_infer_scores"]:
        test_scores = frm.compute_performance_scores(
            y_true=y_label,
            y_pred=y_pred,
            stage="test",
            metric_type=params["metric_type"],
            output_dir=params["output_dir"]
        )

    return True


def main(args):
    """
    Punto de entrada del script de inferencia.

    Inicializa la configuración de inferencia mediante IMPROVE, carga los
    parámetros definidos para DeepTTC y ejecuta la predicción sobre test.
    """

    cfg = DRPInferConfig()

    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="deepttc_params.txt",
        additional_definitions=infer_params
    )

    status = run(params)

    print("\nFinished model inference.")


if __name__ == "__main__":
    main(sys.argv[1:])