using StatisticalRethinking
using JSON
using StanSample
using PSIS
using Statistics

ProjDir = @__DIR__

include(joinpath(ProjDir, "cvit.jl"))

# Data
data = JSON.parsefile(joinpath(ProjDir, "wells.data.json"))
y = Float64.(data["switched"])
x = Float64[data["arsenic"]  data["dist"]]
n, m = size(x)

# Model
model_str = read(open(joinpath(ProjDir, "arsenic_logistic.stan")), String)
tmpdir = joinpath(ProjDir, "tmp")
sm = SampleModel("arsenic_logistic", model_str; tmpdir)

data = (p = m, N = n, y = Int.(y), x = x)
# Fit the model in Stan
rc = stan_sample(sm; data)
nt = read_samples(sm)

# Compute LOO and standard error
log_lik = nt.log_lik'
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

exit()
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

# Compute LOO and standard error
log_lik = sim2[:, names_sim, :]
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
    simcv = stan(stanmodel, standatacv, '.', 
                 CmdStanDir=CMDSTAN_HOME, summary=false)
    ns = filter(x->startswith(x,"log_likt"), simcv.names)
    log_likt = Mamba.combine(simcv[:, ns, :])
    kfcvs[cvitst[cvi]]= PSIS.logsumexp(log_likt) - log(size(log_likt,1))
end

# compare PSIS-LOO and k-fold-CV
p = plot(layer(x = loos, y = kfcvs, Geom.point),
     layer(x = [-3.5,0] ,y=[-3.5,0], Geom.line, style(default_color=colorant"red")),
     Guide.xlabel("PSIS-LOO"),
     Guide.ylabel("10-fold-CV"))

draw(PDF("Compare.pdf", 210mm, 210mm),p) 

