import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


dataset = pd.read_csv("data/data.csv", sep=",")

def label_encoding(df):
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])
    return df

def test_diagnosis_column_():
    dataset = pd.read_csv("data/data.csv", sep=",")
    dataset = label_encoding(dataset)
    assert dataset['diagnosis'].dtypes == 'int'


def test_values_diagnosis_column_():
    dataset = pd.read_csv("data/data.csv", sep=",")
    dataset = label_encoding(dataset)
    assert dataset['diagnosis'].isin([0,1]).all() == True





