data {
  int<lower=0> p;
  int<lower=0> N;
  array[N] int<lower=0,upper=1> y;
  matrix[N,p] x;
  int<lower=0> Nt;
  array[Nt] int<lower=0,upper=1> yt;
  matrix[Nt,p] xt;
}
transformed data {
  matrix[N,p] z;
  matrix[Nt,p] zt;
  vector[p] mean_x;
  vector[p] sd_x;
  for (j in 1:p) { 
    mean_x[j] = mean(col(x,j)); 
    sd_x[j] = sd(col(x,j)); 
    for (i in 1:N)
      z[i,j] = (x[i,j] - mean_x[j]) / sd_x[j]; 
    for (i in 1:Nt)
      zt[i,j] = (xt[i,j] - mean_x[j]) / sd_x[j]; 
  }
}
parameters {
  real beta0;
  vector[p] beta;
  real<lower=0> phi;
}
model {
  vector[N] eta;
  eta = beta0 + z*beta;
  beta0 ~ normal(0, phi);
  beta ~ normal(0, phi);
  phi ~ double_exponential(0, 10);
  y ~ bernoulli_logit(eta);
}

generated quantities {
  vector[N] log_lik;
  vector[Nt] log_likt;
  vector[N] eta;
  vector[Nt] etat;
  eta = beta0 + z*beta;
  etat = beta0 + zt*beta;
  for (i in 1:N)
    log_lik[i] = bernoulli_logit_lpmf(y[i] | eta[i]);
  for (i in 1:Nt)
    log_likt[i] = bernoulli_logit_lpmf(yt[i] | etat[i]);
}

