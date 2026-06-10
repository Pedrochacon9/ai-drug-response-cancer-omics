# python3
# -*- coding:utf-8 -*-

"""
Archivo principal de definición del modelo DeepTTC.

Este script contiene:
- el Dataset de PyTorch utilizado para cargar pares célula-fármaco;
- la rama Transformer encargada de procesar la representación del fármaco;
- la rama MLP encargada de procesar la expresión génica;
- el clasificador final que combina ambas representaciones;
- la clase DeepTTC, que encapsula entrenamiento, validación, predicción,
  guardado y carga del modelo.

En el contexto del TFG, este fichero es importante porque define la arquitectura
principal utilizada para predecir la respuesta farmacológica AUC a partir de:
    representación del fármaco + expresión génica de la línea celular.
"""

import os
import numpy as np
import pandas as pd
import codecs
from sklearn.metrics import mean_squared_error
from lifelines.utils import concordance_index
from scipy.stats import pearsonr, spearmanr
import copy
import time
import pickle
import json

import torch
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from torch import dropout, nn
from torch.utils.data import SequentialSampler

from prettytable import PrettyTable
from subword_nmt.apply_bpe import BPE

# Componentes auxiliares del modelo Transformer.
# Embeddings transforma los tokens del SMILES en vectores.
# Encoder_MultipleLayers aplica varias capas Transformer sobre esos embeddings.
from model_helper import Encoder_MultipleLayers, Embeddings

# Clase encargada de codificar los datos de entrada del modelo,
# especialmente los SMILES de los fármacos.
from Step2_DataEncoding import DataEncoding

from sklearn.metrics import r2_score


class data_process_loader(data.Dataset):
    """
    Dataset personalizado de PyTorch para DeepTTC.

    Cada elemento del dataset corresponde a un par célula-fármaco:
    - v_d: representación/codificación del fármaco;
    - v_p: vector de expresión génica de la línea celular;
    - y: valor de respuesta farmacológica, en este TFG AUC.

    Este Dataset permite usar DataLoader de PyTorch para entrenar y evaluar
    el modelo por lotes.
    """

    def __init__(self, list_IDs, labels, drug_df, rna_df):
        """
        Inicializa el dataset.

        Args:
            list_IDs: índices de las muestras.
            labels: valores reales de respuesta farmacológica.
            drug_df: DataFrame con la información/codificación de fármacos.
            rna_df: DataFrame con la matriz de expresión génica.
        """
        self.labels = labels
        self.list_IDs = list_IDs
        self.drug_df = drug_df
        self.rna_df = rna_df

    def __len__(self):
        """
        Devuelve el número total de muestras del dataset.
        """
        return len(self.list_IDs)

    def __getitem__(self, index):
        """
        Devuelve una muestra concreta del dataset.

        Args:
            index: posición de la muestra dentro del listado.

        Returns:
            tuple: representación del fármaco, expresión génica y etiqueta real.
        """
        index = self.list_IDs[index]

        # Codificación del fármaco generada previamente a partir del SMILES.
        v_d = self.drug_df.iloc[index]['drug_encoding']

        # Vector de expresión génica de la línea celular.
        v_p = np.array(self.rna_df.iloc[index])

        # Valor real de respuesta farmacológica.
        y = self.labels[index]

        return v_d, v_p, y


class transformer(nn.Sequential):
    """
    Rama Transformer del modelo DeepTTC.

    Esta rama procesa la representación tokenizada del SMILES del fármaco.
    Su objetivo es aprender una representación interna del fármaco que capture
    información estructural relevante para la predicción de respuesta.
    """

    def __init__(self, input_dim_drug,
                 transformer_emb_size_drug, dropout,
                 transformer_n_layer_drug,
                 transformer_intermediate_size_drug,
                 transformer_num_attention_heads_drug,
                 transformer_attention_probs_dropout,
                 transformer_hidden_dropout_rate,
                 device):
        """
        Inicializa la rama Transformer.

        Args:
            input_dim_drug: tamaño del vocabulario/tokenización del fármaco.
            transformer_emb_size_drug: dimensión de los embeddings.
            dropout: probabilidad de dropout.
            transformer_n_layer_drug: número de capas Transformer.
            transformer_intermediate_size_drug: dimensión intermedia.
            transformer_num_attention_heads_drug: número de cabezas de atención.
            transformer_attention_probs_dropout: dropout de atención.
            transformer_hidden_dropout_rate: dropout interno del Transformer.
            device: CPU o GPU donde se ejecutará el modelo.
        """
        super(transformer, self).__init__()

        # Capa de embedding que transforma tokens del SMILES en vectores densos.
        self.emb = Embeddings(
            input_dim_drug,
            transformer_emb_size_drug,
            50,
            dropout
        )

        # Encoder Transformer multicapa.
        self.encoder = Encoder_MultipleLayers(
            transformer_n_layer_drug,
            transformer_emb_size_drug,
            transformer_intermediate_size_drug,
            transformer_num_attention_heads_drug,
            transformer_attention_probs_dropout,
            transformer_hidden_dropout_rate
        )

        self.device = device

    def forward(self, v):
        """
        Propagación hacia delante de la rama del fármaco.

        Args:
            v: entrada codificada del fármaco. Incluye tokens y máscara.

        Returns:
            Tensor: representación final del fármaco obtenida por el Transformer.
        """

        # Tokens del SMILES.
        e = v[0].long().to(self.device)

        # Máscara que indica qué posiciones son válidas.
        e_mask = v[1].long().to(self.device)

        # La máscara se adapta al formato esperado por el mecanismo de atención.
        ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
        ex_e_mask = (1.0 - ex_e_mask) * -10000.0

        # Embedding de los tokens.
        emb = self.emb(e)

        # Paso por el encoder Transformer.
        encoded_layers = self.encoder(emb.float(), ex_e_mask.float())

        # Se devuelve la representación asociada al primer token,
        # siguiendo una lógica similar al uso de token resumen en Transformers.
        return encoded_layers[:, 0]


class MLP(nn.Sequential):
    """
    Rama MLP para procesar la expresión génica.

    Esta red recibe como entrada el vector de expresión génica de la línea
    celular y lo transforma en una representación latente de dimensión fija.
    """

    def __init__(self, input_dim, device):
        """
        Inicializa la red MLP de expresión génica.

        Args:
            input_dim: número de variables génicas de entrada.
            device: CPU o GPU donde se ejecutará la red.
        """
        input_dim_gene = input_dim
        hidden_dim_gene = 256

        # Dimensiones internas de la red densa.
        mlp_hidden_dims_gene = [1024, 256, 64]

        super(MLP, self).__init__()

        layer_size = len(mlp_hidden_dims_gene) + 1
        dims = [input_dim_gene] + mlp_hidden_dims_gene + [hidden_dim_gene]

        # Lista de capas lineales consecutivas.
        self.predictor = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)]
        )

        self.device = device

    def forward(self, v):
        """
        Propagación hacia delante de la rama génica.

        Args:
            v: vector de expresión génica.

        Returns:
            Tensor: representación latente de la línea celular.
        """
        v = v.float().to(self.device)

        # Se aplican capas lineales con activación ReLU.
        for i, l in enumerate(self.predictor):
            v = F.relu(l(v))

        return v


class Classifier(nn.Sequential):
    """
    Clasificador final de DeepTTC.

    Combina la representación del fármaco y la representación de la célula.
    A partir de la concatenación de ambas, predice un único valor continuo:
    la respuesta farmacológica AUC.
    """

    def __init__(self, args, model_drug, model_gene):
        """
        Inicializa el clasificador.

        Args:
            args: parámetros del modelo.
            model_drug: rama Transformer del fármaco.
            model_gene: rama MLP de expresión génica.
        """
        super(Classifier, self).__init__()

        self.input_dim_drug = args['input_dim_drug_classifier']
        self.input_dim_gene = args['input_dim_gene_classifier']

        self.model_drug = model_drug
        self.model_gene = model_gene

        self.dropout = nn.Dropout(args['dropout'])

        # Capas densas posteriores a la concatenación.
        self.hidden_dims = [1024, 1024, 512]
        layer_size = len(self.hidden_dims) + 1

        dims = [self.input_dim_drug + self.input_dim_gene] + \
            self.hidden_dims + [1]

        self.predictor = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(layer_size)]
        )

    def forward(self, v_D, v_P):
        """
        Propagación hacia delante del modelo completo.

        Args:
            v_D: entrada del fármaco.
            v_P: entrada de expresión génica.

        Returns:
            Tensor: predicción continua de respuesta farmacológica.
        """

        # Codificación del fármaco mediante Transformer.
        v_D = self.model_drug(v_D)

        # Codificación de la expresión génica mediante MLP.
        v_P = self.model_gene(v_P)

        # Concatenación de ambas representaciones.
        v_f = torch.cat((v_D, v_P), 1)

        # Capas densas finales.
        for i, l in enumerate(self.predictor):
            if i == (len(self.predictor) - 1):
                # Última capa: salida lineal de regresión.
                v_f = l(v_f)
            else:
                # Capas intermedias: dropout + ReLU.
                v_f = F.relu(self.dropout(l(v_f)))

        return v_f


class DeepTTC:
    """
    Clase principal del modelo DeepTTC.

    Esta clase encapsula:
    - construcción de la arquitectura;
    - entrenamiento;
    - validación;
    - predicción;
    - guardado y carga de pesos.

    En el TFG se utiliza para entrenar y evaluar el modelo en los distintos
    escenarios experimentales: split normal, cell-out y drug-out.
    """

    def __init__(self, modeldir, args, gene_dim):
        """
        Inicializa DeepTTC.

        Args:
            modeldir: directorio donde se guardan resultados/modelos.
            args: parámetros de configuración.
            gene_dim: número de variables de expresión génica de entrada.
        """

        # Selección del dispositivo de ejecución.
        # Si hay GPU disponible, se usa CUDA; en caso contrario, CPU.
        devices_list = os.getenv('CUDA_AVAILABLE_DEVICES')

        if not devices_list:
            if args["cuda_name"] != "cuda:0":
                devices_list = args["cuda_name"].split(":")[-1]
            else:
                devices_list = '0'

        self.device = torch.device(
            f'cuda:{devices_list}' if torch.cuda.is_available() else 'cpu'
        )

        # Rama Transformer para el fármaco.
        self.model_drug = transformer(
            args['input_dim_drug'],
            args['transformer_emb_size_drug'],
            args['dropout'],
            args['transformer_n_layer_drug'],
            args['transformer_intermediate_size_drug'],
            args['transformer_num_attention_heads_drug'],
            args['transformer_attention_probs_dropout'],
            args['transformer_hidden_dropout_rate'],
            device=self.device
        )

        self.modeldir = modeldir
        self.record_file = os.path.join(self.modeldir, "valid_markdowntable.txt")
        self.pkl_file = os.path.join(self.modeldir, "loss_curve_iter.pkl")
        self.args = args
        self.gene_dim = gene_dim

        # Rama MLP para expresión génica.
        model_gene = MLP(input_dim=self.gene_dim, device=self.device)

        # Modelo completo: Transformer + MLP + clasificador final.
        self.model = Classifier(self.args, self.model_drug, model_gene)

    def test(self, datagenerator, model):
        """
        Evalúa el modelo sobre un DataLoader.

        Esta función se utiliza tanto durante validación como durante inferencia.
        Calcula predicciones y varias métricas de regresión/correlación.

        Args:
            datagenerator: DataLoader con los datos a evaluar.
            model: modelo PyTorch a evaluar.

        Returns:
            tuple: etiquetas reales, predicciones y métricas calculadas.
        """

        y_label = []
        y_pred = []

        # Modo evaluación: desactiva dropout y otros comportamientos de train.
        model.eval()

        for i, (v_drug, v_gene, label) in enumerate(datagenerator):
            # Predicción del modelo.
            score = model(v_drug, v_gene)

            # Cálculo de pérdida MSE.
            loss_fct = torch.nn.MSELoss()
            n = torch.squeeze(score, 1)

            loss = loss_fct(
                n,
                Variable(torch.from_numpy(np.array(label)).float()).to(self.device)
            )

            # Conversión de predicciones y etiquetas a numpy/listas para métricas.
            logits = torch.squeeze(score).detach().cpu().numpy()
            label_ids = label.to('cpu').numpy()

            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + logits.flatten().tolist()

        # Se devuelve el modelo a modo entrenamiento.
        model.train()

        return y_label, y_pred, \
            mean_squared_error(y_label, y_pred), \
            np.sqrt(mean_squared_error(y_label, y_pred)), \
            pearsonr(y_label, y_pred)[0], \
            pearsonr(y_label, y_pred)[1], \
            spearmanr(y_label, y_pred)[0], \
            spearmanr(y_label, y_pred)[1], \
            concordance_index(y_label, y_pred), \
            r2_score(y_label, y_pred), \
            loss

    def train(self, train_drug, train_rna, val_drug, val_rna):
        """
        Entrena DeepTTC usando train y validación.

        Args:
            train_drug: DataFrame con datos/codificación de fármacos de train.
            train_rna: DataFrame con expresión génica de train.
            val_drug: DataFrame con datos/codificación de fármacos de validación.
            val_rna: DataFrame con expresión génica de validación.

        Returns:
            DeepTTC: objeto con el mejor modelo encontrado durante validación.
        """

        lr = self.args['learning_rate']
        decay = 0
        BATCH_SIZE = self.args['batch_size']
        train_epoch = self.args['epochs']

        self.model = self.model.to(self.device)

        # Optimizador Adam.
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=decay
        )

        loss_history = []

        # Parámetros del DataLoader.
        params = {
            'batch_size': BATCH_SIZE,
            'shuffle': True,
            'num_workers': 0,
            'drop_last': False
        }

        # DataLoader de entrenamiento.
        training_generator = data.DataLoader(
            data_process_loader(
                train_drug.index.values,
                train_drug.Label.values,
                train_drug,
                train_rna
            ),
            **params
        )

        # DataLoader de validación.
        validation_generator = data.DataLoader(
            data_process_loader(
                val_drug.index.values,
                val_drug.Label.values,
                val_drug,
                val_rna
            ),
            **params
        )

        print(training_generator)

        # Se usa el MSE de validación para seleccionar el mejor modelo.
        max_MSE = 1e31
        model_max = copy.deepcopy(self.model)

        valid_metric_record = []
        valid_metric_header = [
            '# epoch', "MSE", 'RMSE',
            "Pearson Correlation", "with p-value",
            'Spearman Correlation', "with p-value2",
            "Concordance Index"
        ]

        scores = {
            'val_loss': 10000000,
            'rmse': 1000000,
            'pcc': 0,
            'scc': 0
        }

        table = PrettyTable(valid_metric_header)

        def float2str(x):
            return '%0.4f' % x

        print('--- Go for Training ---')

        t_start = time.time()
        iteration_loss = 0
        initial_epoch = 0

        # Criterio de parada temprana.
        # Si no mejora el MSE de validación durante varias épocas, se detiene.
        max_iterations_without_improvement = self.args["patience"]
        early_stop_counter = 0
        train_loss = None

        for epo in np.arange(initial_epoch, train_epoch):
            if early_stop_counter >= max_iterations_without_improvement:
                break

            # -------------------------
            # Entrenamiento por lotes
            # -------------------------
            for i, (v_d, v_p, label) in enumerate(training_generator):
                # Predicción del modelo.
                score = self.model(v_d, v_p)

                # Etiqueta real en el dispositivo correspondiente.
                label = Variable(torch.from_numpy(
                    np.array(label))).float().to(self.device)

                # Pérdida MSE, adecuada para regresión de AUC.
                loss_fct = torch.nn.MSELoss()
                n = torch.squeeze(score, 1).float()
                loss = loss_fct(n, label)

                loss_history.append(loss.item())
                iteration_loss += 1

                # Actualización de pesos.
                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss = str(loss.cpu().detach().numpy())[:7]

                if (i % 1000 == 0):
                    t_now = time.time()
                    print(
                        'Training at Epoch ' + str(epo + 1) +
                        ' iteration ' + str(i) +
                        ' with loss ' + train_loss +
                        ". Total time " + str(int(t_now - t_start) / 3600)[:7] + " hours"
                    )

            # -------------------------
            # Validación por época
            # -------------------------
            with torch.set_grad_enabled(False):
                y_true, y_pred, mse, rmse, \
                    pearson, p_val, \
                    spearman, s_p_val, CI, r2, \
                    loss_val = self.test(validation_generator, self.model)

                lst = ["epoch " + str(epo)] + list(map(float2str, [
                    mse, rmse, pearson, p_val, spearman, s_p_val, CI, r2
                ]))

                valid_metric_record.append(lst)

                print(f'Currenf MSE: {mse}')

                # Si mejora el MSE, se guarda una copia del modelo como mejor modelo.
                if mse < max_MSE:
                    early_stop_counter = 0
                    model_max = copy.deepcopy(self.model)
                    max_MSE = mse

                    print(
                        'Validation at Epoch ' + str(epo + 1) +
                        ' with loss:' + str(loss_val.item())[:7] +
                        ', MSE: ' + str(mse)[:7] +
                        ' , Pearson Correlation: ' + str(pearson)[:7] +
                        ' with p-value: ' + str(p_val)[:7] +
                        ' Spearman Correlation: ' + str(spearman)[:7] +
                        ' with p_value: ' + str(s_p_val)[:7] +
                        ' , Concordance Index: ' + str(CI)[:7]
                    )

                    scores['val_loss'] = mse
                    scores['rmse'] = rmse
                    scores['pcc'] = pearson
                    scores['scc'] = spearman
                    scores['r2'] = r2
                    scores['best_epoch'] = epo

                    print(scores)

                else:
                    early_stop_counter += 1

        # Al finalizar, se conserva el mejor modelo según MSE de validación.
        self.model = model_max

        print("\nIMPROVE_RESULT val_loss:\t{}\n".format(scores["val_loss"]))
        print('--- Training Finished ---')

        return self

    def predict(self, drug_data, rna_data):
        """
        Genera predicciones con el modelo entrenado.

        Args:
            drug_data: datos/codificación del fármaco.
            rna_data: matriz de expresión génica.

        Returns:
            tuple: etiquetas reales, predicciones y métricas principales.
        """

        print('predicting...')

        self.model.to(self.device)

        # Dataset de inferencia.
        info = data_process_loader(
            drug_data.index.values,
            drug_data.Label.values,
            drug_data,
            rna_data
        )

        # En predicción no se mezclan las muestras, por eso shuffle=False
        # y se utiliza SequentialSampler.
        params = {
            'batch_size': 16,
            'shuffle': False,
            'num_workers': 8,
            'drop_last': False,
            'sampler': SequentialSampler(info)
        }

        generator = data.DataLoader(info, **params)

        y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI, r2, loss_val = \
            self.test(generator, self.model)

        return y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI

    def save_model(self, model_path):
        """
        Guarda los pesos del modelo entrenado en disco.

        Args:
            model_path: ruta donde se almacenará el fichero del modelo.
        """
        torch.save(self.model.state_dict(), model_path)

    def load_pretrained(self, path):
        """
        Carga los pesos de un modelo previamente entrenado.

        Args:
            path: ruta del fichero de pesos.
        """

        if not os.path.exists(path):
            os.makedirs(path)

        # Carga compatible con GPU o CPU.
        if self.device.type == 'cuda':
            state_dict = torch.load(path)
        else:
            state_dict = torch.load(path, map_location=torch.device('cpu'))

        # Si el modelo fue guardado usando DataParallel, las claves empiezan
        # por 'module.'. Se eliminan esos prefijos para cargarlo correctamente.
        if next(iter(state_dict))[:7] == 'module.':
            from collections import OrderedDict
            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v

            state_dict = new_state_dict

        self.model.load_state_dict(state_dict)

    def preprocess(self, rna_data, drug_data, response_data, response_metric='AUC'):
        """
        Función de preprocesamiento heredada del flujo original de DeepTTC.

        En el TFG, el preprocesamiento principal se realiza en
        deepttc_preprocess_improve.py para adaptarlo a IMPROVE/CSA.
        Esta función se mantiene por compatibilidad con el código original.
        """

        args = self.args

        obj = DataEncoding(
            args['vocab_dir'],
            args['cancer_id'],
            args['sample_id'],
            args['target_id'],
            args['drug_id']
        )

        drug_smiles = drug_data

        drugid2smile = dict(
            zip(drug_smiles['DrugID'], drug_smiles['SMILES'])
        )

        smile_encode = pd.Series(drug_smiles['SMILES'].unique()).apply(
            obj._drug2emb_encoder
        )

        uniq_smile_dict = dict(
            zip(drug_smiles['SMILES'].unique(), smile_encode)
        )

        drug_data.drop(['SMILES'], inplace=True, axis=1)

        drug_data['smiles'] = [
            drugid2smile[i] for i in drug_data['DrugID']
        ]

        drug_data['drug_encoding'] = [
            uniq_smile_dict[i] for i in drug_data['smiles']
        ]

        drug_data = drug_data.reset_index()

        response_data = response_data[['CancID', 'DrugID', response_metric]]
        response_data.columns = ['CancID', 'DrugID', 'Label']

        drug_data = pd.merge(
            response_data,
            drug_data,
            on='DrugID',
            how='inner'
        )

        drug_data.index = range(drug_data.shape[0])
        rna_data.index = range(rna_data.shape[0])

        print('Preprocessing...!!!')
        print(np.shape(rna_data), np.shape(drug_data))

        return rna_data, drug_data


if __name__ == '__main__':

    """
    Bloque de ejecución original del repositorio DeepTTC.

    En el flujo adaptado a IMPROVE/CSA utilizado en el TFG, el entrenamiento
    se lanza mediante deepttc_train_improve.py. Este bloque se conserva como
    referencia del funcionamiento original del modelo.
    """

    # Directorio de vocabulario del modelo original.
    vocab_dir = '.'

    obj = DataEncoding(vocab_dir=vocab_dir)

    # División original de datos por cáncer.
    traindata, testdata = obj.Getdata.ByCancer(random_seed=1)

    # Codificación original de datos.
    traindata, train_rnadata, testdata, test_rnadata = obj.encode(
        traindata=traindata,
        testdata=testdata
    )

    # Construcción y entrenamiento original del modelo.
    modeldir = './Model_80'
    modelfile = modeldir + '/model.pt'

    if not os.path.exists(modeldir):
        os.mkdir(modeldir)

    # En el flujo del TFG se utiliza la clase DeepTTC desde los scripts
    # adaptados a IMPROVE, no desde este bloque principal.
    net = DeepTTC(modeldir=modeldir)
    net = net.train(
        train_drug=traindata,
        train_rna=train_rnadata,
        val_drug=testdata,
        val_rna=test_rnadata
    )

    net.save_model()
    print("Model Saved :{}".format(modelfile))