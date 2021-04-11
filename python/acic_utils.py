import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_p_scores(df, standardize_x = True):
    X = df.copy()
    features_names = X.columns.values[3:]
    X = X[features_names]
    to_categorical = ['S3', 'C1'] # based on output from GAM fit in paper, these are the categorical vars (exclude S3)
    X[to_categorical] = X[['S3', 'C1']].astype('category')
    X = pd.get_dummies(X, columns= to_categorical)
    #nm = X.columns
    X = X.values
    if standardize_x:
        X = StandardScaler().fit_transform(X)
    z= df['Z'].values
    y = df['Y'].values
    
    return X, z, y



def stan_model_summary(stan_fit):
    """
    Parameters
    ----------
    stan_fit : A Stan fit object

    Returns
    -------
    A pandas data frame with summary of posterior parameters.

    """
    summary_dict = stan_fit.summary()
    summary_df = pd.DataFrame(summary_dict['summary'],
                  columns=summary_dict['summary_colnames'],
                  index=summary_dict['summary_rownames'])
    return summary_df
