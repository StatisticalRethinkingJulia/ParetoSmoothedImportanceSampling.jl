# PSIS

[![Build Status](https://travis-ci.org/alvaro1101/PSIS.jl.svg?branch=master)](https://travis-ci.org/alvaro1101/PSIS.jl)

[![Coverage Status](https://coveralls.io/repos/alvaro1101/PSIS.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/alvaro1101/PSIS.jl?branch=master)

[![codecov.io](http://codecov.io/github/alvaro1101/PSIS.jl/coverage.svg?branch=master)](http://codecov.io/github/alvaro1101/PSIS.jl?branch=master)

## Pareto smoothed importance sampling (PSIS) and PSIS leave-one-out cross-validation reference code

### Introduction

These files implement Pareto smoothed importance sampling (PSIS) and
PSIS leave-one-out cross-validation for Julia base in the [Matlab package called PSIS by Aki Vehtari](https://github.com/avehtari/PSIS.git)

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
