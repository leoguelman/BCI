
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

