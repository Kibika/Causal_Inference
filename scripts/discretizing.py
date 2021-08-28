# make outcome column the first column in the dataframe

def discretize_outcome(df, causal_data):
    data_vals = {col: causal_data[col].unique() for col in causal_data.columns}

    diagnosis_map = {v: 'benign' if v == [0]
    else 'malignant' for v in data_vals['diagnosis']}
    df["diagnosis"] = df["diagnosis"].map(diagnosis_map)
    return df


def discretize_independent(df,causal_data):
    data_vals = {col: causal_data[col].unique() for col in causal_data.columns}
    for i in list(discretize_outcome(df, causal_data).columns[1:]):
        map = {v: 'small' if v <= (discretize_outcome(df, causal_data)[str(i)].max() - discretize_outcome(df, causal_data)[str(i)].min()) / 2
        else 'large' for v in data_vals[str(i)]}
        discretize_outcome(df, causal_data)[str(i)] = discretize_outcome(df, causal_data)[str(i)].map(map)
    return df
