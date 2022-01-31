using Distributions
using PDMats # test dependencies
using Test
using Distributed
using Random
using SpecialFunctions
using StatsBase
using LinearAlgebra

import JSON
import ForwardDiff

const tests = [
    "loguniform",
    "arcsine",
    "dirac",
    "truncate",
    "truncnormal",
    "truncated_exponential",
    "truncated_uniform",
    "truncated_discrete_uniform",
    "censored",
    "normal",
    "laplace",
    "cauchy",
    "uniform",
    "lognormal",
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
    "matrixreshaped",
    "matrixvariates",
    "lkjcholesky",
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
    "chernoff",
    "univariate_bounds",
    "negativebinomial",
    "bernoulli",
    "soliton",
    "skewnormal",
    "chi",
    "gumbel",
    "pdfnorm",
    "rician",
    "functionals",
    "density_interface",
    "reshaped",
    "skewedexponentialpower",
    "discreteuniform",
]

printstyled("Running tests:\n", color=:blue)

Random.seed!(345679)

# to reduce redundancy, we might break this file down into seperate `$t * "_utils.jl"` files
include("testutils.jl")

for t in tests
    @testset "Test $t" begin
        include("$t.jl")
    end
end

# print method ambiguities
println("Potentially stale exports: ")
display(Test.detect_ambiguities(Distributions))
println()
