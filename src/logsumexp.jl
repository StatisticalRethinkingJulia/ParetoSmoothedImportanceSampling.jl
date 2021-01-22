"""
    logsumexp(x[, d])

Compute `log(sum(exp(x), d))` of `x`.
"""
function logsumexp(x::AbstractArray, d::Int64=1)
    u = maximum(x)
    abs(u) == Inf && return any(isnan, x) ? T(NaN) : u
    result = u .+ log.(sum(exp.(x .- u), dims=d))
    if length(result) == 1
        result = result[1]
    end
    result
end
