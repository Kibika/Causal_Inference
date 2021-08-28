import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from causalgraphicalmodels import CausalGraphicalModel
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel
from IPython.display import Image
from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
from causalnex.structure.notears import from_pandas, from_pandas_lasso
# import pygraphviz
from causalnex.inference import InferenceEngine
from causalnex.evaluation import roc_auc
from causalnex.evaluation import classification_report
from causalnex.network import BayesianNetwork

from feature_selection import *
from causal_graph import *
from similarity import *
from discretizing import *
from training import *


import dvc
import os
import warnings
import sys
import pathlib

# PATH = pathlib.Path(__file__).parent
# DATA_PATH = PATH.joinpath("./data").resolve()

# path = DATA_PATH.joinpath("result_dataframe.csv")
# repo = 'D:/Stella/Documents/10_Academy/Week_7/causal_inference'
# version = 'v1'
#
# data_url = dvc.api.get_url(path=path,
#                            repo=repo,
#                            rev=version)
# scaler = StandardScaler()
# classifier = RandomForestClassifier(n_estimators=100)
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    dataset = pd.read_csv("data/data.csv", sep=",")
    diagnosis_data = dataset.copy()
    diagnosis_data = diagnosis_data.drop(['Unnamed: 32','id'], axis=1)
    diagnosis_data = label_encoding(diagnosis_data)

    X = diagnosis_data.drop(["diagnosis"], axis=1)
    y = diagnosis_data["diagnosis"]

    X_train, X_test, y_train, y_test = split_data(X, y)

    y_pred = train_predict(X_train, y_train, X_test)

    Accuracy_Score, Precision, Recall = evaluation(y_test, y_pred)

    matrix = confusion_m(y_test, y_pred)
    print(matrix)

    selected_feat = extract_important_features(X_train, y_train)

    # plot causal graph using selected features

    # causal_data = diagnosis_data[list(selected_feat)].copy()
    causal_data = diagnosis_data[['diagnosis','area_mean', 'concavity_mean', 'concave points_mean', 'radius_worst',
       'perimeter_worst', 'area_worst', 'concavity_worst',
       'concave points_worst']]
    initial_graph = graph_lasso_constrained(causal_data)
    print(initial_graph)

    # use domain knowledge to plot final graph
    sm_lasso_constrained = from_pandas_lasso(causal_data, tabu_parent_nodes=['diagnosis'], w_threshold=0.8, beta=0.8)

    sm_lasso_constrained.add_edge("concave points_mean", "diagnosis")
    sm_lasso_constrained.add_edge("concave points_worst", "diagnosis")
    sm_lasso_constrained.add_edge("area_worst", "diagnosis")
    sm_lasso_constrained.add_edge("area_mean", "diagnosis")
    sm_lasso_constrained.add_edge("perimeter_worst", "diagnosis")
    sm_lasso_constrained.add_edge("concavity_mean", "area_mean")

    viz = plot_structure(
        sm_lasso_constrained,
        graph_attributes={"scale": "2.0", 'size': 2.5},
        all_node_attributes=NODE_STYLE.WEAK,
        all_edge_attributes=EDGE_STYLE.WEAK)
    Image(viz.draw(format='png'))

    Image(viz.draw(format='png')).save("causal_graph.png")


    discretised_data = causal_data.copy()
    discretised_data = discretize_outcome(discretised_data)
    discretised_data = discretize_independent(discretised_data)

    model = train(discretised_data, sm_lasso_constrained)
    print(model)
