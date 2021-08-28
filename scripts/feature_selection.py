import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectFromModel

scaler = StandardScaler()
classifier = RandomForestClassifier(n_estimators=100)

def label_encoding(df):
    df['diagnosis'] = le.fit_transform(df['diagnosis'])
    return df

def split_data(X, y):
    # X = df.drop(["diagnosis"], axis=1)
    # y = df["diagnosis"]
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, stratify=y, test_size=0.10, random_state=42
    )
    return X_train, X_test, y_train, y_test

def train_predict(X_train,y_train,X_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    return y_pred

def evaluation(y_test, y_pred):
    Accuracy_Score = accuracy_score(y_test, y_pred)
    Precision = precision_score(y_test, y_pred)
    Recall = recall_score(y_test, y_pred)

    with open("metrics.txt", 'w') as outfile:
        outfile.write("recall: %2.1f%%\n" % Recall)
        outfile.write("precision: %2.1f%%\n" % Precision)
        outfile.write("accuracy: %2.1f%%\n" % Accuracy_Score)
    return Accuracy_Score, Precision, Recall


def confusion_m(y_test,y_pred):
    rf_cnm = confusion_matrix(y_test, y_pred)
    sns.heatmap(rf_cnm / np.sum(rf_cnm), annot=True, fmt='.2%', cmap='Blues')
    ax.xaxis.set_label_position("top")
    # plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion_matrix.png", dpi=120)
    plt.show()


def extract_important_features(X_train,y_train):
    sel = SelectFromModel(RandomForestClassifier(n_estimators=100))
    sel.fit(X_train, y_train)
    sel.get_support()
    selected_feat = pd.DataFrame(X_train).columns[(sel.get_support())]
    return selected_feat

