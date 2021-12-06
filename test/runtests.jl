using ParetoSmoothedImportanceSampling, StanSample
using Test

if haskey(ENV, "JULIA_CMDSTAN_HOME") || haskey(ENV, "CMDSTAN")

    ProjDir = @__DIR__
    include(joinpath(ProjDir, "test_demo_wells.jl"))

else
  println("\nCmdStan or JULIA_CMDSTAN_HOME not set. Skipping tests")
end
