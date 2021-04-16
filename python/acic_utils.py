import pandas as pd
from sklearn.preprocessing import StandardScaler

def pre_process_data(df, standardize_x = True, interactions=False, p_score=None, drop_first=False):
    """
    Parameters
    ----------
    df : A Data Frame.
    standardize_x : bool, optional
        Standardize features by substracting the mean and scaling to unit variance.
    interactions : bool
        Add interactions between each feature and the treatment
        (note: treatment must be named 'Z').
    p_score : 1-D array
        Add propensity score.
    drop_first: bool
        Whether to get k-1 dummies out of k categorical levels by removing the first level.

    Returns
    -------
    X : 2-D array
        Design matrix.
    z : 1-D array
        Treatment indicator.
    y : 1-D array
        Response.
    a_effects : str
        column names in X
    m_effects : str
        column names of main effects
    i_effects: str
        Column names of interaction effects
    
    """
    X = df.copy()
    X = X.iloc[:,3:]
    to_categorical = ['C1', 'XC'] # Similar to Athey/Wager
    X[to_categorical] = X[to_categorical].astype('category')
    X = pd.get_dummies(X, columns= to_categorical, drop_first=drop_first)
    
    if p_score is not None:
        X['p_score'] = p_score
    
    if interactions:
        X['Z'] = df['Z']
        m_effects = X.columns.drop('Z').values.tolist()
        i_effects = ["Z_" + i for i in m_effects]
        X[i_effects] = X[m_effects].multiply(X['Z'], axis="index")
        X = X.drop(columns = ['Z'])
        
    else:
        m_effects = None
        i_effects = None
         
    a_effects = X.columns.tolist()
    X = X.values
    
    if standardize_x:
        X = StandardScaler().fit_transform(X)

    z= df['Z'].values
    y = df['Y'].values
    
    return X, z, y, a_effects, m_effects, i_effects




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
