# PSIS

## Pareto smoothed importance sampling (PSIS) and PSIS leave-one-out cross-validation reference code

### Introduction

These files implement Pareto smoothed importance sampling (PSIS) and
PSIS leave-one-out cross-validation for Julia base on the [Matlab package called `PSIS` by Aki Vehtari](https://github.com/avehtari/PSIS.git)

Included functions
------------------
psisloo
    Pareto smoothed importance sampling leave-one-out log predictive densities.

psislw
    Pareto smoothed importance sampling.

gpdfitnew
    Estimate the paramaters for the Generalized Pareto Distribution (GPD).

gpinv
    Inverse Generalised Pareto distribution function.

sumlogs
    Sum of vector where numbers are represented by their logarithms.

References
----------
Aki Vehtari, Andrew Gelman and Jonah Gabry (2015). Efficient implementation
of leave-one-out cross-validation and WAIC for evaluating fitted Bayesian
models. arXiv preprint arXiv:1507.04544.

Aki Vehtari and Andrew Gelman (2015). Pareto smoothed importance sampling.
arXiv preprint arXiv:1507.02646.

### Corresponding R code

The corresponding R code can be found in [R package called
`loo'](https://github.com/stan-dev/loo) which is also available in CRAN.
                 
### References

- Aki Vehtari, Andrew Gelman and Jonah Gabry (2016). Practical
  Bayesian model evaluation using leave-one-out cross-validation
  and WAIC. Statistics and Computing, [doi:10.1007/s11222-016-9696-4](http://dx.doi.org/10.1007/s11222-016-9696-4). [arXiv preprint arXiv:1507.04544](http://arxiv.org/abs/1507.04544)
- Aki Vehtari, Andrew Gelman and Jonah Gabry (2016). Pareto
  smoothed importance sampling. [arXiv preprint arXiv:1507.02646](http://arxiv.org/abs/1507.02646)
- Jin Zhang & Michael A. Stephens (2009) A New and Efficient
  Estimation Method for the Generalized Pareto Distribution,
  Technometrics, 51:3, 316-325, DOI: 10.1198/tech.2009.08017
