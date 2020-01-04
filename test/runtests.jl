using Distributions
using PDMats # test dependencies
using Test
using Distributed
using Random
using StatsBase
using LinearAlgebra

import JSON
import ForwardDiff

const tests = [
    "truncate",
    "truncnormal",
    "truncated_exponential",
    "normal",
    "mvnormal",
    "mvlognormal",
    "types",
    "utils",
    "samplers",
    "categorical",
    "univariates",
    "continuous",
    "edgecases",
    "fit",
    "multinomial",
    "binomial",
    "poissonbinomial",
    "dirichlet",
    "dirichletmultinomial",
    "logitnormal",
    "mvtdist",
    "kolmogorov",
    "edgeworth",
    "wisharts",
    "matrixbeta",
    "matrixfdist",
    "matrixnormal",
    "matrixtdist",
    "vonmisesfisher",
    "conversion",
    "convolution",
    "mixture",
    "gradlogpdf",
    "noncentralt",
    "locationscale",
    "quantile_newton",
    "semicircle",
    "qq",
    "pgeneralizedgaussian",
    "product",
    "discretenonparametric",
    "functionals",
    "chernoff",
    "univariate_bounds",
    "negativebinomial",
    "bernoulli",
]

printstyled("Running tests:\n", color=:blue)

Random.seed!(345679)

# to reduce redundancy, we might break this file down into seperate `$t * "_utils.jl"` files
include("testutils.jl")

for t in tests
    @testset "Test $t" begin
        Random.seed!(345679)
        include("$t.jl")
    end
end

# print method ambiguities
println("Potentially stale exports: ")
display(Test.detect_ambiguities(Distributions))
println()
