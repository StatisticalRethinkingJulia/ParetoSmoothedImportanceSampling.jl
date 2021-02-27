
"""
    dic(loglike::AbstractVector{<:Real})

Computes Deviance Information Criterion (DIC).

# Arguments
* `log_lik::AbstractArray`: Array of size n x m containing n posterior samples of the log likelihood terms p(y_i|\theta^s).

# Returns
* `dic::Real`: DIC value
"""
function dic(loglike::AbstractVector{<:Real})
    D = deviance.(loglike)
    return mean(D) + 0.5 * var(D)
end

deviance(loglikelihood::Real) = -2 * loglikelihood
