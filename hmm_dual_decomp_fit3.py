#!/usr/bin/python

# stan includes
import pystan
from stan_util import StanModel_cache
import numpy as np

# plotting includes
import arviz
from matplotlib import pyplot as plt
from IPython import embed

# util includes
import sys

hmm_code = """
// simple hmm example (1 output; 2 states)
data {
  int<lower=0> N;
  int<lower=0> K;
  real y[N];
  real mu_lagrange_multiplier[K];
  real theta_lagrange_multiplier[K,K];
}

parameters {
  simplex[K] theta[K];
  // real mu[K];
  positive_ordered[K] mu;
}

model {
  // priors
  target+= normal_lpdf(mu[1] | 3, 1);
  target+= normal_lpdf(mu[2] | 10, 1);
  {
  real mu_acc;
  real theta_acc;

  mu_acc = 0;
  theta_acc = 0;
  for (k in 1:K)
      mu_acc+= mu_lagrange_multiplier[k] * mu[k];
  target+=mu_acc;
  for (k1 in 1:K)
   for (k2 in 1:K)
      theta_acc+= theta_lagrange_multiplier[k1,k2] * theta[k1,k2];
  target+=theta_acc;
  }
  // forward algorithm
  {
  real acc[K];
  real gamma[N, K];
  for (k in 1:K)
    gamma[1, k] = normal_lpdf(y[1] | mu[k], 1);
  for (t in 2:N) {
    for (k in 1:K) {
      for (j in 1:K)
        acc[j] = gamma[t-1, j] + log(theta[j, k]) + normal_lpdf(y[t] | mu[k], 1);
      gamma[t, k] = log_sum_exp(acc);
    }
  }
  target += log_sum_exp(gamma[N]);
  }
}

generated quantities {
  int<lower=1,upper=K> z_star[N];
  real log_p_z_star;
  {
    int back_ptr[N, K];
    real best_logp[N, K];
    for (k in 1:K)
      best_logp[1, k] = normal_lpdf(y[1] | mu[k], 1);
    for (t in 2:N) {
      for (k in 1:K) {
        best_logp[t, k] = negative_infinity();
        for (j in 1:K) {
          real logp;
          logp = best_logp[t-1, j] + log(theta[j, k]) + normal_lpdf(y[t] | mu[k], 1);
          if (logp > best_logp[t, k]) {
            back_ptr[t, k] = j;
            best_logp[t, k] = logp;
          }
        }
      }
    }
    log_p_z_star = max(best_logp[N]);
    for (k in 1:K)
      if (best_logp[N, k] == log_p_z_star)
        z_star[N] = k;
    for (t in 1:(N - 1))
      z_star[N - t] = back_ptr[N - t + 1, z_star[N - t + 1]];
  }
}
"""

# create model
sm = StanModel_cache(model_code=hmm_code)

hmm_data = np.load('hmm_data3.npz')
hmm_data2 = np.load('hmm_data3-2.npz')

K = 3

mu_lagrange_multiplier = np.ones(K)*1e-3
theta_lagrange_multiplier = np.ones([K,K])*1e-3

# fit first chain
y = hmm_data['y']
fit_data = { 'N' : len(y), 'K' : K, 'y' : y, 'mu_lagrange_multiplier' : mu_lagrange_multiplier, 'theta_lagrange_multiplier' : theta_lagrange_multiplier }
fit = sm.sampling(data=fit_data, iter=1000, chains=4)

# fit second chain
y2 = hmm_data2['y']
fit_data2 = { 'N' : len(y2), 'K' : K, 'y' : y2, 'mu_lagrange_multiplier' : -mu_lagrange_multiplier, 'theta_lagrange_multiplier' : -theta_lagrange_multiplier }
fit2 = sm.sampling(data=fit_data2, iter=1000, chains=4)

# dual decomp optimization rate
alpha = -10

def update_lagrange_multipliers(mu_lagrange_multiplier,theta_lagrange_multiplier,fit,fit2,alpha):

    def get_mean_pars(fit):
        pars = fit.extract()
        mu = pars['mu']
        theta = pars['theta']
        return np.mean(mu,axis=0), np.mean(theta,axis=0)

    def get_fit_rhat(fit,pars):
        irhat = fit.summary(pars=pars)['summary_colnames']
        irhat = irhat.index("Rhat")
        rhat = fit.summary(pars=pars)["summary"][:, irhat]
        return rhat

    def fit_rhat_within_parameters(fit,pars,min_rhat=.99,max_rhat=1.01):
        """
        Note, min_rhat and max_rhat taken from stan warning message:
        'WARNING:pystan:Rhat above 1.1 or below 0.9 indicates that the chains very likely have not mixed'
        """
        rhat = get_fit_rhat(fit,pars)
        if np.any(rhat < min_rhat) or np.any(rhat > max_rhat):
            return False
        return True

    mu,theta = get_mean_pars(fit)
    mu2,theta2 = get_mean_pars(fit2)

    mu_diff = mu - mu2
    theta_diff = theta.reshape(K,K) - theta2.reshape(K,K)

    print ('fit rhats:')
    print (get_fit_rhat(fit,pars=['mu','theta']))
    print (get_fit_rhat(fit2,pars=['mu','theta']))

    pars = ['mu','theta']
    if not (fit_rhat_within_parameters(fit,pars) and fit_rhat_within_parameters(fit2,pars)):
        # fit probably did not succeeed, applying lagrange update now would
        # cluster-f things up
        return False

    if np.sum(np.abs(mu_diff)) > 2. or np.sum(np.abs(theta_diff)) > 2.:
        return False

    print('mu_diff abs: ',np.abs(mu_diff), ' sum: ', np.sum(np.abs(mu_diff)))
    print('theta_diff abs: ',np.abs(theta_diff), ' sum: ', np.sum(np.abs(theta_diff)))

    mu_lagrange_multiplier +=  alpha * mu_diff
    theta_lagrange_multiplier +=  alpha * theta_diff
    return True

print('lagrange multipliers before update: ', mu_lagrange_multiplier, theta_lagrange_multiplier)
update_lagrange_multipliers(mu_lagrange_multiplier,theta_lagrange_multiplier,fit,fit2,alpha)
print('lagrange multipliers after update: ', mu_lagrange_multiplier, theta_lagrange_multiplier)

for _ in range(40):
    fit_data = { 'N' : len(y), 'K' : K, 'y' : y, 'mu_lagrange_multiplier' : mu_lagrange_multiplier, 'theta_lagrange_multiplier' : theta_lagrange_multiplier }
    fit = sm.sampling(data=fit_data, iter=1000, chains=2)

    fit_data2 = { 'N' : len(y2), 'K' : K, 'y' : y2, 'mu_lagrange_multiplier' : -mu_lagrange_multiplier, 'theta_lagrange_multiplier' : -theta_lagrange_multiplier }
    fit2 = sm.sampling(data=fit_data2, iter=1000, chains=2)

    print('lagrange multipliers before update: ', mu_lagrange_multiplier, theta_lagrange_multiplier)
    if not update_lagrange_multipliers(mu_lagrange_multiplier,theta_lagrange_multiplier,fit,fit2,alpha):
        pass
    print('lagrange multipliers after update: ', mu_lagrange_multiplier, theta_lagrange_multiplier)

