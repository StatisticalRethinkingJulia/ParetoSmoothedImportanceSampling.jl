
var2(x) = mean(x.^2) .- mean(x)^2

function waic( ll::AbstractArray; pointwise=false , log_lik="log_lik" , kwargs... )
    
    n_samples, n_obs = size(ll)
    #lpd = zeros(n_obs)
    pD = zeros(n_obs)

    lpd = reshape(logsumexp(ll .- log(n_samples); dims=1), n_obs)
    for i in 1:n_obs 
        pD[i] = var2(ll[:,i])
    end

    waic_vec = (-2) .* ( lpd - pD )
    if pointwise == false
        waic = sum(waic_vec)
        lpd = sum(lpd)
        pD = sum(pD)
    else 
        waic = waic_vec
    end

    local se
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
    waic
