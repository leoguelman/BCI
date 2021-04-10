data {
  int<lower=0> N;         // number of data items
  int<lower=0> K;         // number of predictors
  matrix[N, K] x;         // predictor matrix
  vector[N] y;            // outcome vector
  int<lower=0> N_new;     // number of scoring data items
  matrix[N_new, K] x_new; // scoring data matrix
}

parameters {
  real alpha;             // intercept
  vector[K] beta;         // coefficients for predictors
  real<lower=0> sigma;    // error scale
}

model {
  y ~ normal(x * beta + alpha, sigma);  // likelihood
}

generated quantities {
  vector[N_new] prog_scores;
  for (n in 1:N_new)
    prog_scores[n] = normal_rng(x_new[n] * beta, sigma);
}