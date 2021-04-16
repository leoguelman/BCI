###------------------------- Imports

import os
os.chdir('/Users/lguelman/Library/Mobile Documents/com~apple~CloudDocs/LG_Files/Development/BCI/python')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
parameters = {'figure.figsize': (8, 4),
              'font.size': 8, 
              'axes.labelsize': 12}
plt.rcParams.update(parameters)
plt.style.use('fivethirtyeight')

import pystan
import multiprocessing
import stan_utility
import arviz as az

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


import seaborn as sns

from acic_utils import preprocess_prop_score, stan_model_summary
#from sklearn import linear_model
#from sklearn.linear_model import LogisticRegressionCV
#from sklearn.ensemble import RandomForestClassifier
#from stability_selection import StabilitySelection
#import arviz as az
#import xgboost
#import pickle 


###------------------------- Read data 

df = pd.read_csv("../data/synthetic_data.csv")
df
df.info()
df.describe()


X = df.copy()
X = X.iloc[:,3:]
to_categorical = ['C1', 'XC'] # Similar to Athey/Wager
X[to_categorical] = X[to_categorical].astype('category')
X = pd.get_dummies(X, columns= to_categorical)


if interactions:
    X = X[['Z'] + X.columns[3:].tolist()]
    main_cols = X.columns.drop('Z').values.tolist()
    inter_cols = ["Z_" + i for i in main_cols]
    
else:
    X = X.iloc[:,3:]
if p_score is not None:
    X['p_score'] = p_score

###------------------------- Assess whether randomized or observational study
#https://cran.r-project.org/web/packages/MatchIt/vignettes/assessing-balance.html#:~:text=Assessing%20balance%20involves%20assessing%20whether,joint%20distributional%20balance%20as%20well.
#https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3713509/#:~:text=The%20benefit%20of%20the%20prognostic,highly%20predictive%20of%20the%20outcome.
#http://dm.education.wisc.edu/dkaplan2/intellcont/Chen_Kaplan_2015-1.pdf


### Estimate treatment propensity and assess the extent of overlap 

# Pre-Process data 

X, z, _ = preprocess_prop_score(df)


param_grid = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5]
        }

n_folds = 3
param_n_picks = 5

xgb = XGBClassifier(learning_rate=0.01, n_estimators=1000, objective='binary:logistic',
                    silent=True, nthread=1)

skf = StratifiedKFold(n_splits=n_folds, shuffle = True, random_state = 42)

xgb_fits = RandomizedSearchCV(xgb, param_distributions=param_grid,
                              n_iter=param_n_picks, scoring='roc_auc', n_jobs=-1, 
                              cv=skf.split(X,z), verbose=3, random_state=42)

xgb_fits.fit(X, z)

print('\n Best estimator:')
print(xgb_fits.best_estimator_)
print('\n Best AUC score:')
print(xgb_fits.best_score_)
print('\n Best hyperparameters:')
print(xgb_fits.best_params_)

# We now fit the best estimator to all train data 
best_fit = xgb_fits.best_estimator_.fit(X, z)
prop_score = best_fit.predict_proba(X)[:,1]
prop_score  = np.log(prop_score /(1-prop_score))

prop_score_df = pd.DataFrame({'prop_score': prop_score, 'Z':z, 'X1':df['X1'], 'X2': df['X2'],'C1': df['C1'],'S3':df['S3']})

sns.displot(prop_score_df, x="prop_score", hue="Z",  stat="density", common_norm=False)

sns.boxplot(x="C1", y="prop_score", data=prop_score_df)


df.groupby(['S3']).mean()['Z'].plot.line(title="Proportion Treated")

# feature importance
print(best_fit.feature_importances_)

#importance = pd.DataFrame({'imp':best_fit.feature_importances_ , 'names':nm.values})
#importance.sort_values(by = ['imp'])
#best_fit.feature_importances_ 

# plot
from matplotlib import pyplot
pyplot.bar(range(len(best_fit.feature_importances_)), best_fit.feature_importances_)
pyplot.show()

### Bayesian Prognostic scores

X, z, y = preprocess_prop_score(df)

n, p = X[z==1,:].shape

stan_data_dict = {'N': n,
                  'K': p,
                  'x': X[z==1,:],
                  'y': y[z==1],
                  'N_new': X.shape[0],
                  'x_new': X
                  }

sm = pystan.StanModel('../stan/stan_linear_reg.stan') 
multiprocessing.set_start_method("fork", force=True)
fit = sm.sampling(data=stan_data_dict, iter=1000, chains=4)

summary_df = stan_model_summary(fit)
summary_df 

samples = fit.extract(permuted=True)

prog_scores = samples['prog_scores'].T

mcmc_samples = prog_scores.shape[1]
prog_scores_std_diff = np.zeros(mcmc_samples)
prog_scores_diff = np.zeros(mcmc_samples)

for s in range(mcmc_samples):
    prog_scores_diff[s] = np.mean(prog_scores[z==1,s]) - np.mean(prog_scores[z==0,s])
    prog_scores_std_diff[s] = prog_scores_diff[s] / np.std(prog_scores[:,s])
  
                               
# Note: Students with highest potential outcomes under control are more likely to get treatment 
plt.hist(prog_scores_std_diff, bins = 30)
plt.title("Standardized mean difference in Prognostic scores", fontsize=12)
plt.show()    

plt.hist(prog_scores_diff, bins = 30)
plt.title("Mean difference in Prognostic scores", fontsize=12)
plt.show()                              

    




# Compute distribution of standardize mean difference in prognostic scores





###------------------------- Prep and save data for stan

df = pd.read_csv("../data/synthetic_data.csv")
df
df.info()
df.describe()

#for i in range(df.shape[1]):
#    print(df.columns[i], len(np.unique(df.iloc[:,i].values)))

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
