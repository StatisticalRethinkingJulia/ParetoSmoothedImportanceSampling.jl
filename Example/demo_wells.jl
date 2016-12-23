using JSON
using Colors
using Gadfly
using DataFrames

using Mamba
using Stan

using PSIS

# CVIT - Create itr and itst indeces for k-fold-cv
#
#    Description
#     [ITR,ITST]=CVITR(N,K) returns 1xK cell arrays ITR and ITST holding 
#      cross-validation indeces for train and test sets respectively. 
#      K-fold division is balanced with all sets having floor(N/K) or 
#      ceil(N/K) elements.
#
#     [ITR,ITST]=CVITR(N,K,RS) with integer RS>0 also makes random 
#      permutation, using substream RS. This way different permutations 
#      can be produced with different RS values, but same permutation is 
#      obtained when called again with same RS. Function restores the 
#      previous random stream before exiting.
#

# Copyright (c) 2010 Aki Vehtari

# This software is distributed under the GNU General Public
# License (version 2 or later); please refer to the file
# License.txt, included with the software, for details.
    
function cvit(n, k=10, rsubstream=false)

    a = k-rem(n,k)
    b = floor(Int, n/k);

    itst = Any[]
    itr = Any[]

    for cvi in 1:a
        push!(itst, collect(1:b) + (cvi-1) * b)
        push!(itr, setdiff(1:n,itst[cvi])) 
    end
    for cvi in (a+1):k
        push!(itst, (a * b) + collect(1:(b + 1)) + (cvi - a - 1) * (b + 1)) 
        push!(itr, setdiff(1:n,itst[cvi])) 
    end  

    if rsubstream
        rng = MersenneTwister()
        rii = randperm(rng, n)
        for cvi in 1:k
            itst[cvi] = rii[itst[cvi]]
            itr[cvi] = rii[itr[cvi]]
        end
    end
    itr, itst
end


# Data
data = JSON.parsefile("wells.data.json")
y = Float64.(data["switched"])
x = Float64[data["arsenic"]  data["dist"]]
n, m = size(x)

# Model
model_str = readstring(open("arsenic_logistic.stan"))
stanmodel = Stanmodel(name="arsenic_logistic", adapt=500, update=500, model=model_str)

if isfile("sim.jls")
    sim = read("sim.jls", Chains)
else
    standata = [Dict("p" => m, "N" => n, "y" => y, "x" => x)]
    # Fit the model in Stan
    sim = stan(stanmodel, standata, '.', CmdStanDir=CMDSTAN_HOME, summary=false)
    write("sim.jls", sim)
end

r,v,c = size(sim)
ns = filter(x->startswith(x,"log_lik"),sim.names)
log_lik = reshape(permutedims(sim[:,ns,:].value,[3, 1 ,2]), (r*c ,length(ns)))

# Compute LOO and standard error
loo, loos, pk = psisloo(log_lik)
elpd_loo = sum(loos)
se_elpd_loo = std(loos) * sqrt(n)
@printf(">> elpd_loo = %.1f, SE(elpd_loo) = %.1f\n", elpd_loo, se_elpd_loo)

# Check the shape parameter k of the generalized Pareto distribution
if all(pk .< 0.5)
    println("All Pareto k estimates OK (k < 0.5)")
else
    pkn1 = sum((pk .>= 0.5) & (pk .< 1))
    pkn2 = sum(pk .>= 1)
    @printf(">> %d (%.0f%%) PSIS Pareto k estimates between 0.5 and 1\n", pkn1, pkn1/n*100)
    @printf(">> %d (%.0f%%) PSIS Pareto k estimates greater than 1\n", pkn2, pkn2/n*100)
end


# Fit a second model, using log(arsenic) instead of arsenic
x2 = Float64[log.(data["arsenic"])  data["dist"]]

# Model
if isfile("sim2.jls")
    sim2 = read("sim2.jls", Chains)
else
    standata2 = [Dict("p" => m, "N" => n, "y" => y, "x" => x2)]
    # Fit the model in Stan
    sim2 = stan(stanmodel, standata2, '.', CmdStanDir=CMDSTAN_HOME, summary=false)
    write("sim2.jls", sim2)
end

r,v,c = size(sim2)
ns = filter(x->startswith(x,"log_lik"),sim2.names)
log_lik = reshape(permutedims(sim2[:,ns,:].value,[3, 1 ,2]), (r*c ,length(ns)))

# Compute LOO and standard error
loo2, loos2, pk2 = psisloo(log_lik)
elpd_loo = sum(loos2)
se_elpd_loo = std(loos2) * sqrt(n)
@printf(">> elpd_loo = %.1f, SE(elpd_loo) = %.1f\n", elpd_loo, se_elpd_loo)

# Check the shape parameter k of the generalized Pareto distribution
if all(pk .< 0.5)
    println("All Pareto k estimates OK (k < 0.5)")
else
    pkn1 = sum((pk .>= 0.5) & (pk .< 1))
    pkn2 = sum(pk .>= 1)
    @printf(">> %d (%.0f%%) PSIS Pareto k estimates between 0.5 and 1\n", pkn1, pkn1/n*100)
    @printf(">> %d (%.0f%%) PSIS Pareto k estimates greater than 1\n", pkn2, pkn2/n*100)
end

## Compare the models
loodiff = loos - loos2
@printf("elpd_diff = %.1f, SE(elpd_diff) = %.1f\n",sum(loodiff), std(loodiff) * sqrt(n))


## k-fold-CV
# k-fold-CV should be used if several khats>0.5
# in this case it is not needed, but provided as an example
model_str = readstring(open("arsenic_logistic_t.stan"))
stanmodel = Stanmodel(name="arsenic_logistic_t", adapt=500, update=500, model=model_str);

cvitr, cvitst = cvit(n, 10, true)
kfcvs = similar(loos)
for cvi in 1:10
    @printf("%d\n", cvi)

    standatacv = [Dict("p" => m, "N" => length(cvitr[cvi]), "Nt" => length(cvitst[cvi]),
                        "x" => x[cvitr[cvi],:], "y" => y[cvitr[cvi]],
                        "xt" => x[cvitst[cvi],:], "yt" => y[cvitst[cvi]])
                 ]
    # Fit the model in Stan
    simcv = stan(stanmodel, standatacv, '.', CmdStanDir=CMDSTAN_HOME, summary=false)
    r,v,c = size(sim2)
    ns = filter(x->startswith(x,"log_likt"),simcv.names)
    log_likt = reshape(permutedims(simcv[:,ns,:].value,[3, 1 ,2]), (r*c ,length(ns)))
    kfcvs[cvitst[cvi]]= Psis.logsumexp(log_likt) - log(size(log_likt,1))
end

# compare PSIS-LOO and k-fold-CV
p = plot(layer(x = loos, y = kfcvs, Geom.point),
     layer(x = [-3.5,0] ,y=[-3.5,0], Geom.line, style(default_color=colorant"red")),
     Guide.xlabel("PSIS-LOO"),
     Guide.ylabel("10-fold-CV"))

draw(PDF("Compare.pdf", 210mm, 210mm),p) 

