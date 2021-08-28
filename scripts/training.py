from causalnex.inference import InferenceEngine
from causalnex.evaluation import roc_auc
from causalnex.evaluation import classification_report
from causalnex.network import BayesianNetwork


# sm = structured model from causaulnex
# df = data that has been discretised
def train(df, sm_input):
    train, test = train_test_split(df, train_size=0.8, test_size=0.2, random_state=7)
    bn = BayesianNetwork(sm_input)
    bn = bn.fit_node_states(df)
    bn = bn.fit_cpds(train, method="BayesianEstimator", bayes_prior="K2")
    pred = bn.predict(test, 'diagnosis')
    true = np.where(test['diagnosis'] == 'malignant', 1, 0)
    pred = np.where(pred == 'malignant', 1, 0)
    Accuracy_Score = accuracy_score(y_true=true, y_pred=pred)
    Precision = precision_score(y_true=true, y_pred=pred)
    Recall = recall_score(y_true=true, y_pred=pred)
    report = classification_report(bn, test, "diagnosis")
    roc, auc = roc_auc(bn, test, "diagnosis")
    return Accuracy_Score, Precision, Recall, report, auc
