
###------------------------- Todo

#1) Answer question 1 with paper that uses two-step propensity score
#2) Question 2, 3
#3) Complete reading Bayesian NN

###------------------------- Imports

import os
import pystan
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_columns', None)
import seaborn as sns
sns.set_theme()
from sklearn import linear_model
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from stability_selection import StabilitySelection
import arviz as az
import xgboost
import pickle 


os.chdir('/Users/lguelman/Library/Mobile Documents/com~apple~CloudDocs/LG_Files/Development/Bayesian Causal Inference/acic2018')

###------------------------- Read data 

df = pd.read_csv("synthetic_data.csv")
df
df.info()
df.describe()

###------------------------- Assess whether randomized or observational study

## Visual inspection of covariate balance

fig, axes = plt.subplots(2, 5, figsize=(18, 10))
sns.displot(df, x="X1", hue="Z", kind="ecdf", ax = axes[0, 0])
sns.displot(df, x="X2", hue="Z", kind="ecdf", ax = axes[0, 1])
sns.displot(df, x="X3", hue="Z", kind="ecdf", ax = axes[0, 2])
sns.displot(df, x="X4", hue="Z", kind="ecdf", ax = axes[0, 3])
sns.displot(df, x="X5", hue="Z", kind="ecdf", ax = axes[0, 4])
sns.displot(df, x="S3", hue="Z", stat="probability", multiple="dodge", 
            common_norm=False, ax = axes[1, 0])
sns.displot(df, x="C1", hue="Z", stat="probability", multiple="dodge", 
            common_norm=False, ax = axes[1, 1])
sns.displot(df, x="C2", hue="Z", stat="probability", multiple="dodge", 
            common_norm=False, ax = axes[1, 2])
sns.displot(df, x="C3", hue="Z", stat="probability", multiple="dodge", 
            common_norm=False, ax = axes[1, 3])
sns.displot(df, x="XC", hue="Z", stat="probability", multiple="dodge", 
            common_norm=False, ax = axes[1, 4])

plt.show()


## Show proportion treated as a function of S3

df.groupby(['S3']).mean()['Z'].plot.line(title="Proportion Treated")

## Build frequentist ps score model and show overlap 

X = df.copy()
X = X[['S3', 'C1', 'C2', 'C3', 'XC', 'X1', 'X2', 'X3', 'X4', 'X5']]
to_categorical = ['C1'] # based on output from GAM fit in paper, these are the categorical vars (exclude S3)
X[to_categorical] = X[to_categorical].astype('category')
X = pd.get_dummies(X, columns= to_categorical)
features = X.columns
X = X.values
print(X.shape)
y = df['Z'].values
#clf = LogisticRegressionCV(cv=5, random_state=0).fit(X, y)
#clf = RandomForestClassifier().fit(X, y)
#ps = clf.predict_proba(X)[:,1]
model = xgboost.XGBClassifier()
model.fit(X, y)
ps = model.predict_proba(X)[:,1]
#coef = pd.Series(features).to_frame()
#coef['Coef'] = clf.coef_[0]

ps_df = pd.DataFrame({'ps': ps, 'Z':y, 'X1':df['X1'], 'X2': df['X2'],'C1': df['C1'],'S3':df['S3']})
#X1_bin = np.quantile(df['X1'], q = np.arange(0, 1, 0.1))
#ps_df['X1_bin'] = pd.cut(ps_df['X1'], bins=X1_bin)
#X2_bin = np.quantile(df['X2'], q = np.arange(0, 1, 0.1))
#ps_df['X2_bin'] = pd.cut(ps_df['X2'], bins=X2_bin, include_lowest=True)

#ps_df['X1_bins'].value_counts()

sns.displot(ps_df, x="ps", hue="Z",  stat="density", common_norm=False)

#sns.boxplot(x="X1_bin", y="ps", data=ps_df)
#sns.boxplot(x="X2_bin", y="ps", data=ps_df)
#sns.boxplot(x="C1", y="ps", data=ps_df)


###------------------------- Stability Selection

### Prep data 

X = df.copy()
features = ['S3', 'C1', 'C2', 'C3', 'XC', 'X1', 'X2', 'X3', 'X4', 'X5']
X = X[['Z'] + features]
to_categorical = ['C1', 'XC'] # I believe these are the features deemed as categorical in Athey's paper
X[to_categorical] = X[to_categorical].astype('category')
X = pd.get_dummies(X, columns= to_categorical, drop_first=False) #k-1 encoding
main_cols = X.columns.drop('Z').values.tolist()
inter_cols = ["Z_" + i for i in main_cols]
X[inter_cols] = X[main_cols].multiply(X["Z"], axis="index")
#X['ps'] = ps
colnames = X.columns.values
X = X.values
y = df['Y']


selector = StabilitySelection(base_estimator=linear_model.Lasso(random_state=0),lambda_name='alpha',
                              lambda_grid=np.logspace(-5, -1, 50)).fit(X, y)

stab_scores = np.mean(selector.stability_scores_, axis = 1)
stab_scores_df = pd.DataFrame({'colnames':colnames, 'stab_scores':stab_scores})



###------------------------- Prep and save data for stan

df = pd.read_csv("synthetic_data.csv")
df
df.info()
df.describe()

### Prep data 

X = df.copy()
features = ['S3', 'C1', 'C2', 'C3', 'XC', 'X1', 'X2', 'X3', 'X4', 'X5']
X = X[['Z'] + features]
X['ps'] = ps
to_categorical = ['C1', 'XC'] # I believe these are the features deemed as categorical in Athey's paper
X[to_categorical] = X[to_categorical].astype('category')
X = pd.get_dummies(X, columns= to_categorical, drop_first=True) #k-1 encoding
main_cols = X.columns.drop('Z').values.tolist()
inter_cols = ["Z_" + i for i in main_cols]
X[inter_cols] = X[main_cols].multiply(X["Z"], axis="index")
X = X.drop(columns = ['Z'])


X.to_pickle('X')
X = pd.read_pickle('X')

open_file = open("main_cols.pkl", "wb")
pickle.dump(main_cols, open_file)
open_file.close()

open_file = open("inter_cols.pkl", "wb")
pickle.dump(inter_cols, open_file)
open_file.close()


###------------------------- Fit model in stan (Must Restart kernel)


import pystan
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import os
import pandas as pd
import pickle as pkl

os.chdir('/Users/lguelman/Library/Mobile Documents/com~apple~CloudDocs/LG_Files/Development/Bayesian Causal Inference/acic2018')

df = pd.read_csv("synthetic_data.csv")

X = pd.read_pickle('X')

open_file = open("main_cols.pkl", "rb")
main_cols = pkl.load(open_file)
open_file.close()

open_file = open("inter_cols.pkl", "rb")
inter_cols = pkl.load(open_file)
open_file.close()

### Stan code

stan_model1 = """

data {
  int<lower=0> N;                   // sample size
  int<lower=0> N_main_cov;          // number of main effects 
  int<lower=0> N_inter_cov;         // number of interaction covariates
  vector[N] y;                      // observed outcome
  vector[N] z;                      // treatment assigned
  matrix[N, N_main_cov] x;          // covariates
  matrix[N, N_inter_cov] xz_inter;  // interaction terms
  real<lower=-1,upper=1> rho;       // assumed correlation between the potential outcomes
}
parameters {
  real alpha;                       // intercept
  vector[N_main_cov] beta;          // coefficients for x[N]
  vector[N_inter_cov] beta_inter;   // coefficients for xz_inter[N] 
  real tau;                         // super-population average treatment effect
  real<lower=0> sigma_c;            // residual SD for the control
  real<lower=0> sigma_t;            // residual SD for the treated
}
model {
   // PRIORS
   alpha ~ normal(0, 5); 
   beta ~ normal(0, 10);
   beta_inter ~ normal(0, 10);           
   tau ~ normal(0, 5);
   sigma_c ~ normal(0, 5);          
   sigma_t ~ normal(0, 5);

   // LIKELIHOOD
   y ~ normal(alpha + x*beta + xz_inter*beta_inter + tau * z, sigma_t*z + sigma_c*(1-z));
}
generated quantities{
  real tau_fs;                      // finite-sample average treatment effect  
  real y0[N];                       // potential outcome if Z=0
  real y1[N];                       // potential outcome if Z=1
  real tau_unit[N];                 // unit-level treatment effect
  for(n in 1:N){
    real mu_c = alpha + x[n,]*beta;        
    real mu_t = alpha + x[n,]*beta + x[n, ]*beta_inter + tau;
    if(z[n] == 1){                
      y0[n] = normal_rng(mu_c + rho*(sigma_c/sigma_t)*(y[n] - mu_t), sigma_c*sqrt(1 - rho^2)); 
      y1[n] = y[n];
    }else{                        
      y0[n] = y[n];       
      y1[n] = normal_rng(mu_t + rho*(sigma_t/sigma_c)*(y[n] - mu_c), sigma_t*sqrt(1 - rho^2)); 
    }
    tau_unit[n] = y1[n] - y0[n];
  }
  tau_fs = mean(tau_unit);        
}

"""


stan_data1 = {'N': X.shape[0], 
              'y': df['Y'].values,
              'z': df['Z'].values,
              'x': X[main_cols].values,
              'xz_inter': X[inter_cols].values,
              'N_main_cov':X[main_cols].shape[1],
              'N_inter_cov':X[inter_cols].shape[1],
              'rho':0.0}

sm = pystan.StanModel(model_code=stan_model1)
multiprocessing.set_start_method("fork")
fit = sm.sampling(data=stan_data1, iter=1000, chains=4)

   
import arviz as az
inf_data = az.convert_to_inference_data(fit)
az.plot_energy(inf_data)
    
    
#print(fit)
pars = fit.model_pars
print(pars)
summary_dict = fit.summary()
summary_df = pd.DataFrame(summary_dict['summary'], 
                  columns=summary_dict['summary_colnames'], 
                  index=summary_dict['summary_rownames'])

taus = summary_df.loc[['tau_fs', 'tau']]

az.plot_density(inf_data, var_names = ["tau", "tau_fs"], 
                hdi_prob =.95, hdi_markers='v', shade =.5)

summary_df_cov = summary_df.iloc[1:X[main_cols + inter_cols].shape[1]+1,:] 
summary_df_cov = summary_df_cov.rename(index=dict(zip(summary_df_cov.index,
                                                      main_cols + inter_cols)))
summary_df_cov


az.plot_density(inf_data, var_names = ["beta_inter"], figsize=(20, 20),
                hdi_prob =.95, hdi_markers='v', shade =.5)


az.plot_density(inf_data, var_names = [ "beta"], figsize=(20, 20),
                hdi_prob =.95, hdi_markers='v', shade =.5)

### Get estimated individual treatment effect distribution

samples = fit.extract(permuted=True)

tau_unit = samples['tau_unit']
tau_unit.shape

import seaborn as sns
sns.displot(tau_unit[:,0])

tau_unit_mean = np.mean(tau_unit, axis=0)
df['tau_unit_mean'] = tau_unit_mean
tau_unit_mean_bin = np.quantile(df['tau_unit_mean'], q = np.arange(0, 1.1, 0.1))
tau_unit_mean_bin[0] = -2.0
df['tau_unit_mean_bin'] = pd.cut(df['tau_unit_mean'], bins=tau_unit_mean_bin)


profile_by_decile = df.groupby('tau_unit_mean_bin').mean()

profile_by_decile['X1'].plot()
profile_by_decile['X2'].plot()
#profile_by_decile['X3'].plot()
#profile_by_decile['X4'].plot()
profile_by_decile['X5'].plot()
