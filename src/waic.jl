
var2(x) = mean(x.^2) .- mean(x)^2

function log_sum_exp(x) 
    xmax = maximum(x)
    xsum = sum(exp.(x .- xmax))
    xmax + log(xsum)
end

function waic( ll::AbstractArray; pointwise=FALSE , log_lik="log_lik" , kwargs... )
    
    n_samples, n_obs = size(ll)
    lpd <- zeros(n_obs)
    pD <- zeros(n_obs)

    for i in 1:n_obs 
        lpd[i] = log_sum_exp(ll[:,i]) .- log(n_samples)
        pD[i] = var2(ll[:,i])
    end

    waic_vec = (-2) .* ( lpd - pD )
    if pointwise==FALSE
        waic = sum(waic_vec)
        lpd = sum(lpd)
        pD = sum(pD)
    else 
        waic = waic_vec
    end

    try 
        se = sqrt( n_obs*var2(waic_vec) )
    catch e
        prinrln(e)
        se = nothing
    end

    (WAIC=waic, lppd=lpd, penalty=pD, std_err=se)
end

export
    var2,
    log_sum_exp,
    waic
