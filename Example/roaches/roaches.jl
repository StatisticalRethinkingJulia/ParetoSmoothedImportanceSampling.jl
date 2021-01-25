using StatisticalRethinking, StanSample, PSIS
using Printf

ProjDir = @__DIR__

df = CSV.read(joinpath(ProjDir, "roachdata.csv"), DataFrame)
df.roach1 = df.roach1 / 100

roaches1_stan = "
data {
  int<lower=0> N; 
  vector[N] exposure2;
  vector[N] roach1;
  vector[N] senior;
  vector[N] treatment;
  int y[N];
}
transformed data {
  vector[N] log_expo;

  log_expo = log(exposure2);
}
parameters {
  vector[4] beta;
} 
model {
  y ~ poisson_log(log_expo + beta[1] + beta[2] * roach1 + beta[3] * treatment
                  + beta[4] * senior);
}
generated quantities {
  vector[N] log_lik;
  vector[N] eta;
  eta = log_expo + beta[1] + beta[2] * roach1 + beta[3] * treatment
                + beta[4] * senior;
  for (i in 1:N)
    log_lik[i] <- poisson_log_lpmf(y[i] | eta[i]);
}
";

data = (N = size(df, 1), y = Int.(df.y), roach1=df.roach1,
  exposure2=df.exposure2, senior=df.senior, treatment=df.treatment)
n = size(df, 1)

tmpdir=joinpath(ProjDir, "tmp")
sm1 = SampleModel("roaches1", roaches1_stan; tmpdir)
rc1 = stan_sample(sm1; data)

if success(rc1)
  #roaches1_df = read_samples(sm1; output_format=:dataframe)
  #precis(roaches1_df)
  nt1 = read_samples(sm1)

  # Compute LOO and standard error
  log_lik1 = nt1.log_lik'
  loo1, loos1, pk1 = psisloo(log_lik1)
  elpd_loo1 = sum(loos1)
  se_elpd_loo1 = std(loos1) * sqrt(n)
  @printf(">> elpd_loo = %.1f, SE(elpd_loo) = %.1f\n", elpd_loo1, se_elpd_loo1)

  # Check the shape parameter k of the generalized Pareto distribution
  if all(pk1 .< 0.5)
      println("All Pareto k estimates OK (k < 0.5).")
  else
    pk_good = sum(pk1 .<= 0.5)
    pk_ok = length(pk1[pk1 .<= 0.7]) - pk_good
    pk_bad = length(pk1[pk1 .<= 1]) - pk_good - pk_ok
    println((good=pk_good, ok=pk_ok, bad=pk_bad, very_bad=sum(pk1 .> 1)))
  end
end

begin
  scatter(pk1, xlab="Datapoint", ylab="Pareto shape k",
    marker=2.5, lab="Pk points")
  hline!([0.5], lab="pk = 0.5");hline!([0.7], lab="pk = 0.7")
  hline!([1], lab="pk = 1.0")
  title!("PSIS diagnostic plot")
end