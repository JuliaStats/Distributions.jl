using Distributions
using JSON, ForwardDiff, Calculus, PDMats # test dependencies
using Test
using Distributed
using Random
using StatsBase

tests = [
    "types",
    "utils",
    "samplers",
    "categorical",
    "univariates",
    "continuous",
    "fit",
    "multinomial",
    "binomial",
    "poissonbinomial",
    "dirichlet",
    "dirichletmultinomial",
    "mvnormal",
    "mvlognormal",
    "mvtdist",
    "kolmogorov",
    "edgeworth",
    "matrix",
    "vonmisesfisher",
    "conversion",
    "mixture",
    "gradlogpdf",
    "truncate",
    "noncentralt",
    "locationscale",
    "quantile_newton",
    "semicircle",
    "qq",
    "truncnormal",
]

printstyled("Running tests:\n", color=:blue)

using Random
Random.seed!(345679)

res = map(tests) do t
    @eval module $(Symbol("Test_", t))
    using Distributions
    using JSON, ForwardDiff, Calculus, PDMats # test dependencies
    using Test
    using Random
    Random.seed!(345679)
    using LinearAlgebra
    using StatsBase
    include($t * ".jl")
    end
    return
end

# print method ambiguities
println("Potentially stale exports: ")
display(Test.detect_ambiguities(Distributions))
println()
