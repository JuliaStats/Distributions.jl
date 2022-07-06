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
    "univariate/continuous/loguniform",
    "univariate/continuous/arcsine",
    "univariate/discrete/dirac",
    "truncate",
    "truncated/normal",
    "truncated/exponential",
    "truncated/uniform",
    "truncated/discrete_uniform",
    "censored",
    "univariate/continuous/normal",
    "univariate/continuous/laplace",
    "univariate/continuous/cauchy",
    "univariate/continuous/uniform",
    "univariate/continuous/lognormal",
    "multivariate/mvnormal",
    "multivariate/mvlognormal",
    "types", # extra file compared to /src
    "utils",
    "samplers",
    "univariate/discrete/categorical",
    "univariates",
    "univariate/continuous", # extra file compared to /src
    "edgecases", # extra file compared to /src
    "fit", # extra file compared to /src
    "multivariate/multinomial",
    "univariate/discrete/binomial",
    "univariate/discrete/betabinomial",
    "univariate/discrete/poissonbinomial",
    "multivariate/dirichlet",
    "multivariate/dirichletmultinomial",
    "univariate/continuous/logitnormal",
    "multivariate/mvtdist",
    "univariate/continuous/kolmogorov",
    "edgeworth",
    "matrixreshaped", # extra file compared to /src
    "matrixvariates",
    "cholesky/lkjcholesky",
    "multivariate/vonmisesfisher",
    "conversion",
    "convolution",
    "mixture", # extra file compared to /src
    "gradlogpdf", # extra file compared to /src
    "univariate/continuous/noncentralt",
    "univariate/locationscale",
    "quantile_newton", # extra file compared to /src
    "univariate/continuous/semicircle",
    "qq",
    "univariate/continuous/pgeneralizedgaussian",
    "multivariate/product",
    "univariate/discrete/discretenonparametric",
    "univariate/continuous/chernoff",
    "univariate_bounds", # extra file compared to /src
    "univariate/discrete/negativebinomial",
    "univariate/discrete/bernoulli",
    "univariate/discrete/soliton",
    "univariate/continuous/skewnormal",
    "univariate/continuous/chi",
    "univariate/continuous/gumbel",
    "pdfnorm",
    "univariate/continuous/rician",
    "functionals",
    "density_interface",
    "reshaped",
    "univariate/continuous/skewedexponentialpower",
    "univariate/discrete/discreteuniform",
]

printstyled("Running tests:\n", color=:blue)

Random.seed!(345679)

# to reduce redundancy, we might break this file down into seperate `$t * "_utils.jl"` files
include("testutils.jl")

@testset "Distributions" begin
    for t in tests
        @testset "Test $t" begin
            include("$t.jl")
        end
    end
end

# print method ambiguities
println("Potentially stale exports: ")
display(Test.detect_ambiguities(Distributions))
println()
