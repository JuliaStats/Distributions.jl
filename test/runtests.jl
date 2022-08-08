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
    "product",
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

    ### missing files compared to /src:
    # "common",
    # "estimators",
    # "genericfit",
    # "matrix/inversewishart",
    # "matrix/lkj",
    # "matrix/matrixbeta",
    # "matrix/matrixfdist",
    # "matrix/matrixnormal",
    # "matrix/matrixtdist",
    # "matrix/wishart",
    # "mixtures/mixturemodel",
    # "mixtures/unigmm",
    # "multivariate/mvnormalcanon",
    # "multivariate/product",
    # "quantilealgs",
    # "samplers/aliastable",
    # "samplers/binomial",
    # "samplers/categorical",
    # "samplers/discretenonparametric",
    # "samplers/exponential",
    # "samplers/gamma",
    # "samplers/multinomial",
    # "samplers/obsoleted",
    # "samplers/poisson",
    # "samplers/poissonbinomial",
    # "samplers/vonmises",
    # "samplers/vonmisesfisher",
    # "show",
    # "truncated/loguniform",
    # "univariate/continuous/beta",
    # "univariate/continuous/beta",
    # "univariate/continuous/betaprime",
    # "univariate/continuous/biweight",
    # "univariate/continuous/chisq",
    # "univariate/continuous/cosine",
    # "univariate/continuous/epanechnikov",
    # "univariate/continuous/erlang",
    # "univariate/continuous/exponential",
    # "univariate/continuous/fdist",
    # "univariate/continuous/frechet",
    # "univariate/continuous/gamma",
    # "univariate/continuous/generalizedextremevalue",
    # "univariate/continuous/generalizedpareto",
    # "univariate/continuous/inversegamma",
    # "univariate/continuous/inversegaussian",
    # "univariate/continuous/ksdist",
    # "univariate/continuous/ksonesided",
    # "univariate/continuous/levy",
    # "univariate/continuous/logistic",
    # "univariate/continuous/noncentralbeta",
    # "univariate/continuous/noncentralchisq",
    # "univariate/continuous/noncentralf",
    # "univariate/continuous/normalcanon",
    # "univariate/continuous/normalinversegaussian",
    # "univariate/continuous/pareto",
    # "univariate/continuous/rayleigh",
    # "univariate/continuous/studentizedrange",
    # "univariate/continuous/symtriangular",
    # "univariate/continuous/tdist",
    # "univariate/continuous/triangular",
    # "univariate/continuous/triweight",
    # "univariate/continuous/weibull",
    # "univariate/continuous/noncentralf",
    # "univariate/discrete/geometric",
    # "univariate/discrete/hypergeometric",
    # "univariate/discrete/noncentralhypergeometric",
    # "univariate/discrete/poisson",
    # "univariate/discrete/skellam",

    ### file is present but was not included in list
    # "multivariate_stats", # extra file compared to /src
    # "univariate/continuous/vonmises",
]

printstyled("Running tests:\n", color=:blue)

Random.seed!(345679)

# to reduce redundancy, we might break this file down into seperate `$t * "_utils.jl"` files
include("testutils.jl")

@testset "Distributions" begin
    @testset "Test $t" for t in tests
        include("$t.jl")
    end
end

# print method ambiguities
println("Potentially stale exports: ")
display(Test.detect_ambiguities(Distributions))
println()
