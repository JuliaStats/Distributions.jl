using Distributions
using JSON, ForwardDiff, Calculus, PDMats # test dependencies
using Test
using Distributed
using Random
using StatsBase

tests = [
    "mvnormal",
    "mvlognormal",
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
    "product",
    "truncnormal",
    "truncated_exponential",
    "discretenonparametric",
    "functionals", 
    "chernoff"
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
    include("testutils.jl") # to reduce redundancy, we might break this file down into seperate `$t * "_utils.jl"` files
    include($t * ".jl")
    end
    return
end

# print method ambiguities
println("Potentially stale exports: ")
display(Test.detect_ambiguities(Distributions))
println()
