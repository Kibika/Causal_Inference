# make outcome column the first column in the dataframe

def discretize_outcome(df):
    data_vals = {col: causal_data[col].unique() for col in causal_data.columns}

    diagnosis_map = {v: 'benign' if v == [0]
    else 'malignant' for v in data_vals['diagnosis']}
    df["diagnosis"] = df["diagnosis"].map(diagnosis_map)
    return df


def discretize_independent(df):
    for i in list(discretised_data.columns[1:]):
        map = {v: 'small' if v <= (discretised_data[str(i)].max() - discretised_data[str(i)].min()) / 2
        else 'large' for v in data_vals[str(i)]}
        discretised_data[str(i)] = discretised_data[str(i)].map(map)
    return df
