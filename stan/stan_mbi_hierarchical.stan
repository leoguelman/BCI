
data {
  int<lower=0> N;                   // sample size
  int<lower=0> J;                   // number of schools
  int<lower=1,upper=J> school[N];   // school data
  int<lower=0> N_s;                 // number of school-level effects
  int<lower=0> N_sz;                // number of school-level interaction effects
  int<lower=0> N_ns;                // number of individual-level effects
  int<lower=0> N_nsz;               // number of individual-level interaction effects
  matrix[N, N_s] x_s;               // school-level covariates
  matrix[N, N_sz] x_sz;             // school-level covariates x Z interaction
  matrix[N, N_ns] x_ns;             // individual-level covariates
  matrix[N, N_nsz] x_nsz;           // individual-level covariates x Z interaction
  vector[N] y;                      // observed outcome
  vector[N] z;                      // treatment assigned
  real<lower=-1,upper=1> rho;       // assumed correlation between the potential outcomes
}

parameters {
  vector[J] alpha;                  // intercepts
  vector[N_s] beta_s;               // coefficients for x_s[N]
  vector[N_sz] beta_sz;             // coefficients for x_sz[N] 
  vector[N_ns] beta_ns;             // coefficients for x_ns[N]
  vector[N_nsz] beta_nsz;           // coefficients for x_nsz[N] 
  real tau;                         // super-population average treatment effect
  real<lower=0> sigma_c;            // residual SD for the control
  real<lower=0> sigma_t;            // residual SD for the treated
  real mu_alpha;                    // intercept global mean 
  real<lower=0> sigma_alpha;        // intercept variance
}

transformed parameters {
  vector[N] y_hat;
  vector[N] m;

  for (n in 1:N) {
  m[n] = alpha[school[n]] + x_s[n] * beta_s + x_sz[n] * beta_sz;
  y_hat[n] = m[n] + x_ns[n] * beta_ns + x_nsz[n] * beta_nsz + tau * z[n];
  }

}

model {
   // PRIORS
   mu_alpha ~ normal(0, 1);
   alpha ~ normal(mu_alpha, sigma_alpha); 
   beta_s ~ normal(0, 10);
   beta_sz ~ normal(0, 10);       
   beta_ns ~ normal(0, 10);
   beta_nsz ~ normal(0, 10);
   tau ~ normal(0, 5);
   sigma_c ~ normal(0, 5);          
   sigma_t ~ normal(0, 5);

   // LIKELIHOOD
 
   y ~ normal(y_hat, sigma_t*z + sigma_c*(1-z));
}

generated quantities{
  real tau_fs;                      // finite-sample average treatment effect  
  real y0[N];                       // potential outcome if Z=0
  real y1[N];                       // potential outcome if Z=1
  real tau_unit[N];                 // unit-level treatment effect
  for(n in 1:N){
    real mu_c = alpha[school[n]] + x_s[n,] * beta_s + x_ns[n,] * beta_ns;        
    real mu_t = alpha[school[n]] + x_s[n,] * beta_s + x_s[n,] * beta_sz + x_ns[n,] * beta_ns + x_ns[n,] * beta_nsz + tau;
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

