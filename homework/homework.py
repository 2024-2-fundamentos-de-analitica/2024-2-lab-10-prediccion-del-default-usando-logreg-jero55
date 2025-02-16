# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


import gzip
import json
import os

import pickle
import zipfile
from glob import glob
import pandas as pd  
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline 

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def load_data(input_directory):
    dfs = []
    routes = glob(f"{input_directory}/*")
    for route in routes:
        with zipfile.ZipFile(f"{route}", mode="r") as z:
            for file in z.namelist():

                with z.open(file) as f:
                    dfs.append(pd.read_csv(f, sep=",", index_col=0))

    return dfs


def create_directory(output_directory):
    if os.path.exists(output_directory):
        for file in glob(f"{output_directory}/*"):
            os.remove(file)
        os.rmdir(output_directory)
    os.makedirs(output_directory)


def clean_df(df):
    df = df.copy()
    df = df.rename(columns={"default payment next month": "default"})
    df = df.loc[df["MARRIAGE"] != 0]
    df = df.loc[df["EDUCATION"] != 0]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x >= 4 else x)
    return df.dropna()


def split_data(df):
    return df.drop(columns=["default"]), df["default"]


def pipeline():
    cat_features = ["SEX", "EDUCATION", "MARRIAGE"]
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ],
        remainder=MinMaxScaler(),
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("feature_selection", SelectKBest(score_func=f_classif, k=10)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def estimator(pipeline):
    param_grid = {
        "feature_selection__k": range(1, 30),
        "classifier__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "classifier__solver": ["liblinear"],
    }

    return GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        n_jobs=-1,
        verbose=2,
        refit=True,
    )


def save_model(path, estimator):
    create_directory("files/models/")

    with gzip.open(path, "wb") as f:
        pickle.dump(estimator, f)


def metrics(dataset_type, y_true, y_pred):
    return {
        "type": "metrics",
        "dataset": dataset_type,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def confusion(dataset_type, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset_type,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
    }


def main():
    test_df, train_df = [clean_df(df) for df in load_data("files/input")]

    x_train, y_train = split_data(train_df)
    x_test, y_test = split_data(test_df)

    pipeline = pipeline()

    estimator = estimator(pipeline)
    estimator.fit(x_train, y_train)

    save_model(
        os.path.join("files/models/", "model.pkl.gz"),
        estimator,
    )

    create_directory("files/output/metrics/")

    y_test_pred = estimator.predict(x_test)
    test_precision_metrics = metrics("test", y_test, y_test_pred)
    y_train_pred = estimator.predict(x_train)
    train_precision_metrics = metrics("train", y_train, y_train_pred)

    test_confusion_metrics = confusion("test", y_test, y_test_pred)
    train_confusion_metrics = confusion("train", y_train, y_train_pred)

    with open("files/output/metrics.json", "w", encoding="utf-8") as file:
        file.write(json.dumps(train_precision_metrics) + "\n")
        file.write(json.dumps(test_precision_metrics) + "\n")
        file.write(json.dumps(train_confusion_metrics) + "\n")
        file.write(json.dumps(test_confusion_metrics) + "\n")


if __name__ == "__main__":
    main()