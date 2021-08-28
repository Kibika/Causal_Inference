#outlier detection and replacement by median
def outlier(df):
  column_name=list(df.columns[2:])
  for i in column_name:
    upper_quartile=df[i].quantile(0.75)
    lower_quartile=df[i].quantile(0.25)
    df[i]=np.where(df[i]>upper_quartile,df[i].mean(),np.where(df[i]<lower_quartile,df[i].median(),df[i]))
  return df