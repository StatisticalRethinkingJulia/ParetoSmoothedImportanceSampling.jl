using StatisticalRethinking, StanSample, PSIS, RDatasets

df = RDatasets.dataset("datasets", "cars")

cars_stan = "
data {
    int N;
    vector[N] speed;
    vector[N] dist;
}
parameters {
    real a;
    real b;
    real sigma;
}
transformed parameters{
    vector[N] mu;
    mu = a + b * speed;
}
model {
    a ~ normal(0, 100);
    b ~ normal(0, 10);
    sigma ~ exponential(1);
    dist ~ normal(mu, sigma)    ;
}
generated quantities {
    vector[N] log_lik;
    for (i in 1:N)
        log_lik[i] = normal_lpdf(dist[i] | mu[i], sigma);
}
"

cars_stan_model = SampleModel("cars.model", cars_stan)
data = (N = size(df, 1), speed = df.Speed, dist = df.Dist)
rc = stan_sample(cars_stan_model; data)

if success(rc)
    cars_df = read_samples(cars_stan_model; output_format=:dataframe)
    precis(cars_df[:, [:a, :b, :sigma]])

    nt_cars = read_samples(cars_stan_model);
end

log_lik = nt_cars.log_lik'
ns, n = size(log_lik)
lppd = [log_sum_exp(log_lik[:, i] .- log(ns)) for i in 1:n]
pwaic = [var(log_lik[:, i]) for i in 1:n]
-2(sum(lppd) - sum(pwaic)) |> display

waic(log_lik) |> display
