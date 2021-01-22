"""Pareto smoothed importance sampling (PSIS)

This module implements Pareto smoothed importance sampling (PSIS) and PSIS
leave-one-out cross-validation for Python (Numpy).

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
"""
module PSIS

export psislw
export psisloo


# Copyright (c) 2015 Aki Vehtari, Tuomas Sivula
# Original Matlab version by Aki Vehtari. Translation to Python
# by Tuomas Sivula.

# This software is distributed under the GNU General Public
# License (version 3 or later); please refer to the file
# License.txt, included with the software, for details.

"""
    psisloo(log_lik, [wcpp, wtrunc])

PSIS leave-one-out log predictive densities.

Computes the log predictive densities given posterior samples of the log
likelihood terms p(y_i|\theta^s) in input parameter `log_lik`. Returns a
sum of the leave-one-out log predictive densities `loo`, individual
leave-one-out log predictive density terms `loos` and an estimate of Pareto
tail indeces `ks`. If tail index k>0.5, variance of the raw estimate does
not exist and if tail index k>1 the mean of the raw estimate does not exist
and the PSIS estimate is likely to have large variation and some bias.

# Arguments
* `log_lik::Union{AbstractArray, Mamba.Chains}`: Array of size n x m containing n posterior samples of the log likelihood terms p(y_i|\theta^s).
* `wcpp::Real`: Percentage of samples used for GPD fit estimate (default is 20).
* `wtrunc::Float64`: Positive parameter for truncating very large weights to n^wtrunc. Providing False or 0 disables truncation. Default values is 3/4.

# Returns
* `loo::Real`: sum of the leave-one-out log predictive densities.
* `loos::AbstractArray`: Individual leave-one-out log predictive density terms.* `ks::AbstractArray`: Estimated Pareto tail indeces.
"""
# Compute LOO and standard error
function psisloo(log_lik::AbstractArray, wcpp::Int64=20, wtrunc::Float64=3/4)
   
    # log raw weights from log_lik
    lw = copy(log_lik)

    # compute Pareto smoothed log weights given raw log weights
    lwp, ks = psislw(-lw, wcpp, wtrunc)

    lwp += lw
    loos = logsumexp(lwp, 1)
    loo = sum(loos)

    return loo, loos, ks
end

"""
    psislw(lw, [wcpp, wtrunc])

Compute the Pareto smoothed importance sampling (PSIS).

# Arguments
* `lw::Union{AbstractArray, Mamba.Chains}`: Array of size n x m containing m sets of n log weights. It is also possible to provide one dimensional array of length n.
* `wcpp::Real`: Percentage of samples used for GPD fit estimate (default is 20).
* `wtrunc::Float64`: Positive parameter for truncating very large weights to n^wtrunc. Providing False or 0 disables truncation. Default values is 3/4.

# Returns
* `lw_out::AbstractArray`: Smoothed log weights
* `kss::AbstractArray`: Pareto tail indices
"""
function psislw(lw::AbstractArray, wcpp::Int64=20, wtrunc::Float64=3/4)

    lw_out = copy(lw)

    if ~(1 <= ndims(lw_out) <= 2)
        throw(DimensionMismatch("Argument `lw` must be 1 or 2 dimensional."))
    end
    if size(lw_out,1) <= 1
        error("More than one log-weight needed.")
    end
    return _psislw(lw_out, wcpp, wtrunc)
end

function _psislw(lw_out::Array{Float64}, wcpp::Int64, wtrunc::Float64)

    if ndims(lw_out) == 2
        n, m = size(lw_out)
    elseif ndims(lw_out) == 1
        n = length(lw_out)
        m = 1
    end
    kss = zeros(Float64, m)

    # precalculate constants
    cutoffmin = log(realmin(Float64)) 
    logn = log(n)

    # loop over sets of log weights
    for i in 1:m
        x = view(lw_out, :, i)
        x .-= maximum(x)
        xcutoff = max(quantile(x, 1 - wcpp/100), cutoffmin)
        expxcutoff = exp(xcutoff)
        x2 = x[x .> xcutoff]
        n2 = length(x2)
        if n2 <= 4
            # not enough tail samples for gpdfitnew
            k = Inf
        else
            # order of tail samples
            x2si = sortperm(x2)
            # fit generalized Pareto distribution to the right tail samples
            x2 .= exp.(x2) .- expxcutoff
            k, sigma = gpdfitnew(x2)
            # compute ordered statistic for the fit
            sti = collect(0.5:n2) / n2 
            qq = gpinv(sti, k, sigma)
            qq .= log.(qq .+ expxcutoff)
            # place the smoothed tail into the output array
            x[x .> xcutoff] = qq[x2si]
        end
        if wtrunc > 0
            # truncate too large weights
            lwtrunc = wtrunc * logn - logn + logsumexp(x)
            x[x .> lwtrunc] = lwtrunc
        end
        # renormalize weights
        x .-= logsumexp(x)
        # store tail index k
        kss[i] = k
    end
    # If the provided input array is one dimensional, return kss as scalar.
    if ndims(lw_out) == 1
        kss = kss[1]
    end
    return lw_out, kss
end

"""
    gpdfitnew(x)

Estimate the paramaters for the Generalized Pareto Distribution (GPD). Returns empirical Bayes estimate for the parameters of the two-parameter generalized Parato distribution given the data.

# Arguments
* `x::AbstractArray`: One dimensional data array.

# Returns
* `k::Float64, sigma::Float64`: Estimated parameter values.

# Notes
* This function returns a negative of Zhang and Stephens's k, because it is more common parameterization.
"""
function gpdfitnew(x::AbstractArray{T}) 
    if ndims(x) != 1 || length(x) <= 1
        throw(DimensionMismatch())
    end
    sort!(x)

    n = length(x)
    m = 80 + trunc(Int, sqrt(n))
    prior = 3
    
    bs = collect(1:m) - 0.5  
    bs .= (1 .- sqrt.(m ./ bs)) ./ prior ./ x[trunc(Int64, n/4 + 0.5)] .+ 1 / x[end]

    ks = mean(log1p.(-bs .* x'), 2)[:,1]
    L = n * (log.(bs ./ -ks) .- ks .- 1)
    w = similar(L)
    w .= (1 ./ sum(exp.(L .- L'), 1)')[:,1]

    # remove negligible weights
    dii = w .>= (10 * eps(Float64))
    if ~all(dii)
        w = w[dii]
        bs = bs[dii]
    end
    # normalise w
    w ./= sum(w)
    # posterior mean for b
    b = sum(bs .* w)
    # Estimate for k, note that we return a negative of Zhang and
    # Stephens's k, because it is more common parameterisation.
    k = mean(log1p.(-b .* x))
    # estimate for sigma
    sigma = -k / b

    return k, sigma

end

"""
    gpinv(p, k, sigma)

Inverse Generalised Pareto distribution function.
"""
function gpinv(p, k, sigma)
    x = similar(p)
    if sigma <= 0
        return x
    end
    ok = (p .> 0) & (p .< 1)
    pv = view(p, ok)
    xv = view(x,ok)
    if abs(k) < eps(Float64)
        xv .= -log1p.(-pv)
    else
        xv .= expm1.(log1p.(-pv) .* (-k)) ./ k
    end
    x .*= sigma
    if ~all(ok)
        x[p .== 0] = 0
        if k >= 0
            x[p .== 1] = Inf
        else
            x[p .== 1] = -sigma / k
        end
    end
    x
end

"""
    logsumexp(x[, d])

Compute `log(sum(exp(x), d))` of `x`.
"""
function logsumexp{T<:Real}(x::AbstractArray{T}, d::Int64=1)
    u = maximum(x)
    abs(u) == Inf && return any(isnan, x) ? T(NaN) : u
    result = u .+ log.(sum(exp.(x-u), d))
    if length(result) == 1
        result = result[1]
    end
    result
end


end
