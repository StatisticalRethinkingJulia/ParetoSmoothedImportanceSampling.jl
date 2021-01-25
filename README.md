# PSIS

| **Project Status**                                                               |  **Build Status**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
|![][project-status-img] | ![][CI-build] |


### Purpose of this package

This Julia package implements Pareto smoothed importance sampling (PSIS) and
PSIS leave-one-out cross-validation based on the [Matlab package called `PSIS` by Aki Vehtari](https://github.com/avehtari/PSIS.git).

### Installation

Once registered, PSIS.jl can be installed with:
```
Pkg.dev("PSIS")
```
Usually I have only a few packages `permanently` installed, e.g.:
```
(@v1.6) pkg> st
      Status `~/.julia/environments/v1.6/Project.toml`
  [634d3b9d] DrWatson v1.16.6
  [44cfe95a] Pkg
```
To use the demonstration Pluto notebooks, you can add:
```
  [c3e4b0f8] Pluto v0.12.18
  [7f904dfe] PlutoUI v0.6.11
```

### Included functions

psisloo
    Pareto smoothed importance sampling leave-one-out log predictive densities.

psislw
    Pareto smoothed importance sampling.

gpdfitnew
    Estimate the paramaters for the Generalized Pareto Distribution (GPD).

gpinv
    Inverse Generalised Pareto distribution function.

logsumexp
    Sum of a vector where numbers are represented by their logarithms.

### Acknowledgements

The Julia translation has been done by ... ( @alvaro1101 on Github ).

### Corresponding R code

The corresponding R code can be found in [R package called
`loo`](https://github.com/stan-dev/loo) which is also available in CRAN.
                 
### References

- Aki Vehtari, Andrew Gelman and Jonah Gabry (2016). Practical
  Bayesian model evaluation using leave-one-out cross-validation
  and WAIC. Statistics and Computing, [doi:10.1007/s11222-016-9696-4](http://dx.doi.org/10.1007/s11222-016-9696-4). [arXiv preprint arXiv:1507.04544](http://arxiv.org/abs/1507.04544)
- Aki Vehtari, Andrew Gelman and Jonah Gabry (2016). Pareto
  smoothed importance sampling. [arXiv preprint arXiv:1507.02646](http://arxiv.org/abs/1507.02646)
- Jin Zhang & Michael A. Stephens (2009) A New and Efficient
  Estimation Method for the Generalized Pareto Distribution,
  Technometrics, 51:3, 316-325, DOI: 10.1198/tech.2009.08017


[CI-build]: https://github.com/goedman/PSIS.jl/workflows/CI/badge.svg?branch=master

[issues-url]: https://github.com/goedman/PSIS.jl/issues

[project-status-img]: https://img.shields.io/badge/lifecycle-wip-orange.svg
