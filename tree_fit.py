#!/usr/bin/python

# stan includes
import pystan
from stan_util import StanModel_cache
import numpy as np

# plotting includes
import arviz
from matplotlib import pyplot as plt
from IPython import embed

tree_code = """
// belief propagation on a tree
data {
  int<lower=0> N;
  int<lower=0> K;
  int<lower=0> D;
  real y[N];
  int<lower=-1> A[N,D]; // adjacency (outbound)
  int<lower=0> C[N]; // degree (outbound)
  int<lower=0> O[N]; // visit order
  real<lower=0.1> sigma;
  real mu_guess[K];
  real theta_guesses[K,K];
  int<lower=0> which_priors; // flag indicating which priors to use
  // description list of `which_priors`:
  // 0 - no priors
  // 1 - only mu priors
  // 2 - only theta priors
  // 3 - both mu and theta priors
}

parameters {
  simplex[K] theta[K];
  positive_ordered[K] mu;
}

model {
  // priors - probably important for stability
  if ( which_priors == 1 || which_priors == 3 ) {
    for (k in 1:K)
      target+= normal_lpdf(mu[k] | mu_guess[k], 1);
  }
  if ( which_priors == 2 || which_priors == 3 ) {
    for (k1 in 1:K)
      for (k2 in 1:K)
        target+= normal_lpdf(theta[k1,k2] | theta_guesses[k1,k2], 1);
  }

  // belief propagation on a tree
  {
  real acc[K];
  real gamma[N, K];
  int current_node;
  int neighbour_node;
  int current_node_degree;

  // unknown init
  current_node = 1;
  for (k in 1:K)
    gamma[current_node, k] = normal_lpdf(y[current_node] | mu[k], sigma);

  // biased init towards 2
  //current_node = 1;
  //for (k in 1:K)
  //  gamma[current_node, k] = normal_lpdf(mu[2] | mu[k], sigma);

  for (t in 1:N) {
    current_node = O[t]+1; // 1 based indexing
    current_node_degree = C[current_node];
    // check for leaf node
    if ( current_node_degree == 0 ) {
      target += log_sum_exp(gamma[current_node]);
    } else {
      for (n in 1:current_node_degree) {
        //print(">>attempting index n:", n);
        neighbour_node = A[current_node,n]+1; // 1 based indexing
        for (k in 1:K) {
          for (j in 1:K)
            acc[j] = gamma[current_node, j] + log(theta[j, k]) + normal_lpdf(y[neighbour_node] | mu[k], sigma);
          gamma[neighbour_node, k] = log_sum_exp(acc);
        }
      }
    }
  }
  }
}
"""

#TODO: add generation of samples using variation of verterbi algorithm

show_trace = False

tree_data = np.load('tree_data.npz')

y = tree_data['y']
z = tree_data['z']
A = tree_data['A']
D = tree_data['D']
C = tree_data['C']
O = tree_data['O']
sigma = tree_data['sigma']

fit_data = {'N': len(y), 'K': 2, 'y': y, 'A': A, 'D': D, 'C': C, 'sigma': sigma, 'O': O, 'mu_guess': [
    20., 80.], 'theta_guesses': [[.6, .4], [.4, .6]], 'which_priors': 2}

sm = StanModel_cache(model_code=tree_code)
fit = sm.sampling(data=fit_data, iter=8000, chains=1)

if show_trace:
    arviz.plot_trace(fit)
    plt.show()

print(pystan.stansummary(fit, pars=['mu', 'theta']))

pars = fit.extract()
mu = pars['mu']
theta = pars['theta']
