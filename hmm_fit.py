#!/usr/bin/python

# stan includes
import pystan
from stan_util import StanModel_cache
import numpy as np

# plotting includes
import arviz
from matplotlib import pyplot as plt
from IPython import embed

hmm_code = """
// simple hmm example (1 output; 2 states)
data {
  int<lower=0> N;
  int<lower=0> K;
  real y[N];
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

hmm_data = np.load('hmm_data.npz')

y = hmm_data['y']
z = hmm_data['z']
fit_data = { 'N' : len(y), 'K' : 2, 'y' : y }

sm = StanModel_cache(model_code=hmm_code)
fit = sm.sampling(data=fit_data, iter=1000, chains=4)

print(pystan.stansummary(fit,pars=['mu','theta']))

pars = fit.extract()

mu = pars['mu']
theta = pars['theta']
z_star = pars['z_star']

z_prime = mu[np.arange(mu.shape[0]),(z_star-1).T.astype('int')].T
y_hat = z_prime + np.random.randn(*z_prime.shape)*.6

plt.figure()
plt.plot(y_hat[0].T,'pink',linewidth=.5,label='y-hat (predicted)')
plt.plot(y_hat[1:].T,'pink',linewidth=.5,label='_nolegend_')
plt.plot(y,'black',linewidth=2,label='y (observed)')
plt.legend()
plt.show()

