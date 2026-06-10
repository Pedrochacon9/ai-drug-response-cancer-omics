import sys
from pathlib import Path
from typing import Dict

# Librerías generales del sistema y de tratamiento de datos.
# Se utilizan para gestionar rutas, ejecutar comandos externos,
# guardar objetos auxiliares y manipular tablas numéricas.
import subprocess
import joblib
import pandas as pd
import numpy as np
import os

# Clase propia/original del pipeline DeepTTC encargada de codificar
# la información molecular del fármaco a partir de su SMILES.
from Step2_DataEncoding import DataEncoding

# Escaladores de scikit-learn empleados para normalizar las variables ómicas.
# La normalización se ajusta sobre train y posteriormente se aplica a val/test.
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler

# Importaciones principales de IMPROVE.
# DRPPreprocessConfig permite inicializar la configuración estándar
# para tareas de Drug Response Prediction.
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
from improvelib.utils import str2bool
import improvelib.utils as frm

# Utilidades específicas de IMPROVE para cargar datos de fármacos,
# datos ómicos y respuestas farmacológicas.
import improvelib.applications.drug_response_prediction.drug_utils as drugs_utils
import improvelib.applications.drug_response_prediction.omics_utils as omics_utils
import improvelib.applications.drug_response_prediction.drp_utils as drp

# Definición adicional de parámetros específicos del preprocesamiento.
from model_params_def import preprocess_params


# Ruta base del script. IMPROVE la utiliza como referencia para encontrar
# ficheros de configuración y recursos auxiliares del modelo.
filepath = Path(__file__).resolve().parent


def process_gene_expression(data, gene_expression_columns, dtype=None):
    """
    Extrae las columnas de expresión génica y, opcionalmente, reduce
    la precisión numérica para disminuir el uso de memoria.

    En este TFG, esta función es útil porque la matriz de expresión génica
    puede contener muchas variables y se guarda posteriormente en formato HDF5.

    Args:
        data (pd.DataFrame): DataFrame completo tras combinar célula, fármaco y respuesta.
        gene_expression_columns (list): Columnas correspondientes a expresión génica.
        dtype (str, optional): Tipo numérico deseado. Puede ser 'float32' o 'float16'.
                               Si es None, se mantiene el tipo original.

    Returns:
        pd.DataFrame: Matriz de expresión génica procesada.
    """
    df_gene_expression = data[gene_expression_columns]

    # Si se especifica un tipo numérico, se convierte la matriz.
    # Esto permite reducir memoria, especialmente al trabajar con matrices grandes.
    if dtype is not None:
        if dtype not in ["float32", "float16"]:
            raise ValueError("dtype must be 'float32' or 'float16'")
        df_gene_expression = df_gene_expression.astype(dtype)

    return df_gene_expression


def preprocess(args, rna_data, drug_data, response_data, response_metric='AUC'):
    """
    Combina los datos de expresión génica, fármacos y respuesta farmacológica.

    Esta función adapta los datos al formato esperado por DeepTTC:
    - la célula se representa mediante expresión génica;
    - el fármaco se representa mediante SMILES codificado;
    - la respuesta se almacena como etiqueta 'Label'.

    Args:
        args (dict): Parámetros del pipeline.
        rna_data (pd.DataFrame): Datos de expresión génica de las líneas celulares.
        drug_data (pd.DataFrame): Datos de fármacos y sus SMILES.
        response_data (pd.DataFrame): Pares célula-fármaco con respuesta asociada.
        response_metric (str): Métrica de respuesta utilizada como variable objetivo.

    Returns:
        tuple: rna_data y drug_data preparados para generar los ficheros finales.
    """

    # Se crea el objeto de codificación de DeepTTC.
    # Este objeto transforma los SMILES en representaciones internas que
    # posteriormente utilizará la rama del fármaco del modelo.
    obj = DataEncoding(
        args,
        args["input_supp_data_dir"],
        args["canc_col_name"],
        args["sample_col_name"],
        args["y_col_name"],
        args["drug_col_name"]
    )

    drug_smiles = drug_data

    # Diccionario que asocia cada identificador de fármaco con su SMILES.
    drugid2smile = dict(
        zip(drug_smiles[args["drug_col_name"]], drug_smiles['SMILES'])
    )

    # Codificación de cada SMILES único.
    # Se calcula una vez por molécula para evitar repetir trabajo innecesario.
    smile_encode = pd.Series(drug_smiles['SMILES'].unique()).apply(
        obj._drug2emb_encoder
    )

    # Diccionario SMILES -> codificación.
    uniq_smile_dict = dict(
        zip(drug_smiles['SMILES'].unique(), smile_encode)
    )

    # Se añade a cada fármaco su SMILES y su codificación correspondiente.
    drug_data['smiles'] = [
        drugid2smile[i] for i in drug_data[args["drug_col_name"]]
    ]

    drug_data['drug_encoding'] = [
        uniq_smile_dict[i] for i in drug_data['smiles']
    ]

    drug_data = drug_data.reset_index()

    # Se conserva únicamente la información necesaria de respuesta:
    # identificador de célula, identificador de fármaco y métrica objetivo.
    response_data = response_data[[
        args["canc_col_name"],
        args["drug_col_name"],
        response_metric
    ]]

    # DeepTTC espera que la respuesta se llame 'Label'.
    response_data.columns = [
        args["canc_col_name"],
        args["drug_col_name"],
        'Label'
    ]

    # Se une la respuesta con la información del fármaco.
    # La unión se hace por identificador de fármaco.
    drug_data = pd.merge(
        response_data,
        drug_data,
        on=args["drug_col_name"],
        how='inner'
    )

    drug_data.index = range(drug_data.shape[0])
    rna_data.index = range(rna_data.shape[0])

    print('Preprocessing...!!!')
    print(np.shape(rna_data), np.shape(drug_data))

    return rna_data, drug_data


def build_common_data(params: Dict):
    """
    Carga los datos comunes a todas las particiones: fármacos y expresión génica.

    En el flujo IMPROVE/CSA, las respuestas se cargan por separado para cada
    partición train/val/test, pero los datos de entrada de fármacos y células
    son comunes. Esta función prepara esas dos fuentes principales.

    Args:
        params (Dict): Diccionario de parámetros de IMPROVE y del modelo.

    Returns:
        tuple:
            df_drug: DataFrame con identificadores de fármaco y SMILES.
            gene_expression: DataFrame con expresión génica de líneas celulares.
    """

    # Cargadores de IMPROVE para datos ómicos y datos de fármacos.
    omics_loader = omics_utils.OmicsLoader(params)
    drugs_loader = drugs_utils.DrugsLoader(params)

    # Carga de la matriz de expresión génica.
    # Este fichero contiene las líneas celulares y sus variables génicas.
    gene_expression = omics_loader.dfs['cancer_gene_expression.tsv']

    # Carga del fichero de SMILES.
    # Se adapta el nombre de columnas para que coincida con lo que espera DeepTTC.
    df_drug = drugs_loader.dfs['drug_SMILES.tsv']
    df_drug = df_drug.reset_index()
    df_drug.columns = [params["drug_col_name"], "smiles"]

    params['drug_id'] = params["drug_col_name"]
    df_drug["SMILES"] = df_drug["smiles"]

    def gene_selection(df, genes_fpath, canc_col_name):
        """
        Selecciona únicamente los genes incluidos en el fichero de genes landmark.

        En algunos experimentos se reduce la matriz de expresión génica a un
        subconjunto de genes relevantes, por ejemplo los genes landmark de LINCS.

        Args:
            df (pd.DataFrame): Matriz de expresión génica.
            genes_fpath (Path): Ruta al fichero con genes seleccionados.
            canc_col_name (str): Nombre de la columna identificadora de célula.

        Returns:
            pd.DataFrame: Matriz de expresión génica filtrada.
        """
        with open(genes_fpath) as f:
            genes = [str(line.rstrip()) for line in f]

        # Se conservan solo los genes presentes realmente en el DataFrame.
        genes = sorted(list(set(genes).intersection(set(df.columns[1:]))))
        cols = [canc_col_name] + genes

        return df[cols]

    # Si el parámetro use_lincs está activo, se filtra la expresión génica.
    if params["use_lincs"]:
        genes_fpath = filepath / "landmark_genes"
        gene_expression = gene_selection(
            gene_expression,
            genes_fpath,
            canc_col_name=params["canc_col_name"]
        )

    return df_drug, gene_expression


def _download_default_dataset(default_data_url):
    """
    Descarga un dataset por defecto desde una URL.

    Esta función forma parte del flujo general del modelo, aunque en este TFG
    normalmente se ha trabajado con los datos ya disponibles en la estructura
    IMPROVE/CSA local.
    """
    url = default_data_url
    improve_data_dir = '.'

    if improve_data_dir is None:
        improve_data_dir = '.'

    OUT_DIR = improve_data_dir
    print('outdir after: {}'.format(OUT_DIR))

    url_length = len(url.split('/')) - 5

    if not os.path.isdir(OUT_DIR):
        os.mkdir(OUT_DIR)

    url = url.strip('\'')

    try:
        subprocess.run(['rm', '*index*'])
    except:
        pass

    command = [
        'wget',
        '--recursive',
        '--no-clobber',
        '-nH',
        f'--cut-dirs={url_length}',
        '--no-parent',
        f'--directory-prefix={OUT_DIR}',
        f'{url}'
    ]

    subprocess.run(command)

    try:
        subprocess.run(['rm', '*index*'])
    except:
        pass


def download_model_data(params):
    """
    Descarga los datos por defecto del modelo si se requiere.
    """
    _download_default_dataset(params["default_data_url"])


def download_dataset(params):
    """
    Descarga el benchmark CSA desde el repositorio público de CANDLE/IMPROVE.

    En el flujo final del TFG, esta descarga puede no ejecutarse si los datos
    ya están disponibles localmente en la carpeta csa_data.
    """
    mainpath = Path(os.environ["IMPROVE_DATA_DIR"])

    command = (
        f'wget --directory-prefix={mainpath} --cut-dirs=7 -nH -np -m '
        f'ftp://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/'
        f'single_drug_drp/benchmark-data-pilot1/csa_data'
    )

    tokens = command.split(' ')
    subprocess.run(tokens)


def prepare_dataframe(args, gene_expression, smiles, responses):
    """
    Construye el DataFrame final combinando expresión génica, SMILES y respuesta.

    Esta función prepara una tabla unificada a partir de:
    - matriz de expresión génica;
    - datos de fármaco y codificación SMILES;
    - respuestas AUC para pares célula-fármaco.

    Args:
        args (dict): Parámetros del pipeline.
        gene_expression (pd.DataFrame): Matriz de expresión génica.
        smiles (pd.DataFrame): Datos de fármacos y SMILES.
        responses (pd.DataFrame): Respuesta farmacológica.
    
    Returns:
        tuple:
            data: DataFrame combinado.
            gene_expression_columns: columnas de expresión génica.
            drug_columns: columnas asociadas al fármaco y a la respuesta.
    """

    gene_expression, drug_data = preprocess(
        args,
        gene_expression,
        smiles,
        responses,
        args["y_col_name"]
    )

    # Se elimina la columna de índice generada durante el reset_index anterior.
    drug_data = drug_data.drop(['index'], axis=1)

    # Columnas asociadas al fármaco y a la respuesta.
    drug_columns = [
        x for x in drug_data.columns
        if x not in [args["canc_col_name"], args["drug_col_name"]]
    ]

    # Unión de expresión génica y datos de fármaco/respuesta por identificador de célula.
    data = pd.merge(
        gene_expression,
        drug_data,
        on=args["canc_col_name"],
        how='inner'
    )

    # Se separan las columnas de expresión génica para guardarlas posteriormente.
    gene_expression = gene_expression.drop([args["canc_col_name"]], axis=1)
    gene_expression_columns = gene_expression.columns

    return data, gene_expression_columns, drug_columns


def get_common_samples(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        ref_col: str):
    """
    Filtra dos DataFrames para conservar únicamente las muestras comunes.

    En este TFG se utiliza para asegurar que las respuestas farmacológicas
    solo se mantienen cuando existe también información ómica disponible
    para la línea celular correspondiente.

    Args:
        df1 (pd.DataFrame): Primer DataFrame, normalmente respuestas.
        df2 (pd.DataFrame): Segundo DataFrame, normalmente expresión génica.
        ref_col (str): Columna usada para buscar elementos comunes.

    Returns:
        tuple: df1 y df2 filtrados con los mismos identificadores comunes.
    """
    common_ids = list(set(df1[ref_col]).intersection(df2[ref_col]))

    df1 = df1[df1[ref_col].isin(common_ids)].reset_index(drop=True)
    df2 = df2[df2[ref_col].isin(common_ids)].reset_index(drop=True)

    return df1, df2


def scale_df(dataf, scaler_name="std", scaler=None, verbose=False):
    """
    Escala las variables numéricas de un DataFrame.

    La normalización se ajusta sobre el conjunto de entrenamiento y se aplica
    después a validación y test. Esto evita que la información de validación
    o test influya en el cálculo de los parámetros de escalado.

    Args:
        dataf (pd.DataFrame): DataFrame a escalar.
        scaler_name (str): Tipo de escalado. Opciones: 'std', 'minmax',
                           'minabs', 'robust' o 'none'.
        scaler: Escalador ya ajustado. Si es None, se crea uno nuevo.
        verbose (bool): Si True, muestra mensajes adicionales.

    Returns:
        tuple:
            dataf: DataFrame escalado.
            scaler: Escalador utilizado.
    """

    # Si no se desea escalado, se devuelve el DataFrame original.
    if scaler_name is None or scaler_name == "none":
        if verbose:
            print("Scaler is None (no df scaling).")
        return dataf, None

    # Se seleccionan únicamente columnas numéricas.
    df_num = dataf.select_dtypes(include="number")

    # Si no se proporciona un escalador, se crea y se ajusta.
    # Esto ocurre en el conjunto de entrenamiento.
    if scaler is None:
        if scaler_name == "std":
            scaler = StandardScaler()
        elif scaler_name == "minmax":
            scaler = MinMaxScaler()
        elif scaler_name == "minabs":
            scaler = MaxAbsScaler()
        elif scaler_name == "robust":
            scaler = RobustScaler()
        else:
            print(
                f"The specified scaler {scaler_name} is not implemented (no df scaling)."
            )
            return dataf, None

        df_norm = scaler.fit_transform(df_num)

    # Si el escalador ya existe, se aplica directamente.
    # Esto ocurre en validación y test.
    else:
        df_norm = scaler.transform(df_num)

    # Se sustituyen las columnas numéricas por sus valores escalados.
    dataf[df_num.columns] = df_norm

    return dataf, scaler


def build_stage_dependent_data(params: Dict,
                               stage: str,
                               df_drug: pd.DataFrame,
                               df_cell_all: pd.DataFrame,
                               scaler):
    """
    Construye y guarda los datos procesados para una partición concreta.

    Esta función se ejecuta tres veces: train, val y test. Para cada partición:
    - carga las respuestas correspondientes;
    - filtra las células con expresión génica disponible;
    - escala la expresión génica;
    - combina célula, fármaco y respuesta;
    - guarda los datos X en HDF5;
    - guarda las etiquetas Y en CSV.

    Args:
        params (Dict): Parámetros de IMPROVE/DeepTTC.
        stage (str): Partición a procesar: 'train', 'val' o 'test'.
        df_drug (pd.DataFrame): Datos de fármacos y SMILES.
        df_cell_all (pd.DataFrame): Matriz completa de expresión génica.
        scaler: Escalador usado para normalizar variables ómicas.

    Returns:
        scaler: Escalador ajustado sobre train y reutilizado en val/test.
    """

    # Asociación entre nombre de partición y fichero de split correspondiente.
    stages = {
        "train": params["train_split_file"],
        "val": params["val_split_file"],
        "test": params["test_split_file"]
    }

    # Limpieza de comillas en parámetros tipo string.
    for key in params:
        if type(params[key]) == str:
            params[key] = params[key].strip('"')

    # Carga de las respuestas de la partición actual.
    # El resultado incluye pares célula-fármaco y métricas como AUC.
    df_response = drp.DrugResponseLoader(
        params,
        split_file=stages[stage],
        verbose=False
    ).dfs["response.tsv"]

    # Se conservan solo las respuestas para las que existe expresión génica.
    df_y, df_cell = get_common_samples(
        df1=df_response,
        df2=df_cell_all,
        ref_col=params["canc_col_name"]
    )

    print(df_y[[params["canc_col_name"], params["drug_col_name"]]].nunique())

    # En train se ajusta el escalador y se guarda.
    if stage == "train":
        df_cell, scaler = scale_df(df_cell, scaler_name=params["scaling"])

        if params["scaling"] is not None and params["scaling"] != "none":
            scaler_fname = os.path.join(
                params["output_dir"],
                "cell_xdata_scaler.gz"
            )

            joblib.dump(scaler, scaler_fname)
            print("Scaling object created is stored in: ", scaler_fname)

    # En validación y test se reutiliza el escalador ya ajustado en train.
    else:
        df_cell, _ = scale_df(df_cell, scaler=scaler)

    # Se reduce la tabla de respuesta a las columnas necesarias.
    df_y = df_y[[
        params["drug_col_name"],
        params["canc_col_name"],
        params["y_col_name"]
    ]]

    # Se construye el DataFrame final para esta partición.
    data, gene_expression_columns, drug_columns = prepare_dataframe(
        params,
        df_cell,
        df_drug,
        df_y
    )

    # Separación de las dos entradas principales del modelo:
    # expresión génica y datos/codificación del fármaco.
    df_gene_expression = process_gene_expression(
        data,
        gene_expression_columns,
        params["gene_dtype"]
    )

    df_drug = data[drug_columns]

    # Ruta de salida del fichero HDF5 de la partición.
    out_path = os.path.join(
        params["output_dir"],
        frm.build_ml_data_file_name(params["data_format"], stage=stage)
    )

    print(out_path)

    # Se guardan las matrices X en un HDF5:
    # - 'drug': información/codificación del fármaco;
    # - 'gene_expression': matriz de expresión génica.
    df_output = {
        'drug': df_drug,
        'gene_expression': df_gene_expression
    }

    for key in df_output:
        df_output[key].to_hdf(out_path, key)

    # Guardado de las etiquetas Y y los identificadores asociados.
    data[params['y_col_name']] = data['Label']

    y_df = pd.DataFrame(
        data[[
            'Label',
            params['y_col_name'],
            params['canc_col_name'],
            params['drug_col_name']
        ]]
    )

    frm.save_stage_ydf(y_df, stage, params['output_dir'])

    return scaler


def run(params: Dict):
    """
    Ejecuta el preprocesamiento completo.

    Primero carga los datos comunes de fármacos y expresión génica.
    Después procesa de forma secuencial las particiones train, val y test.

    Args:
        params (dict): Parámetros de IMPROVE y DeepTTC.

    Returns:
        str: Directorio donde se guardan los ficheros procesados.
    """

    df_drug, df_cell_all = build_common_data(params)

    stages = ["train", "val", "test"]
    scaler = None

    for st in stages:
        print(f"Building stage: {st}")

        scaler = build_stage_dependent_data(
            params,
            st,
            df_drug,
            df_cell_all,
            scaler
        )

    return params["output_dir"]


def main(args):
    """
    Punto de entrada del script.

    Inicializa la configuración de preprocesamiento mediante IMPROVE,
    carga los parámetros definidos para DeepTTC y ejecuta el flujo completo.
    """

    cfg = DRPPreprocessConfig()

    params = cfg.initialize_parameters(
        filepath,
        default_config="deepttc_params.txt",
        additional_definitions=preprocess_params
    )

    # Las funciones de descarga se mantienen disponibles, pero en el flujo
    # principal del TFG se trabaja con los datos ya disponibles localmente.
    # download_model_data(params)
    # download_dataset(params)

    ml_data_outdir = run(params)

    print("\nFinished data preprocessing.")


if __name__ == "__main__":
    main(sys.argv[1:])