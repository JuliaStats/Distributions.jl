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
    "aqua",
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
    "univariate/continuous/normalcanon",
    "univariate/continuous/laplace",
    "univariate/continuous/cauchy",
    "univariate/continuous/uniform",
    "univariate/continuous/lognormal",
    "multivariate/mvnormal",
    "multivariate/mvlogitnormal",
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
    "namedtuple/productnamedtuple",
    "univariate/discrete/discretenonparametric",
    "univariate/continuous/chernoff",
    "univariate_bounds", # extra file compared to /src
    "univariate/discrete/negativebinomial",
    "univariate/discrete/geometric",
    "univariate/discrete/bernoulli",
    "univariate/discrete/poisson",
    "univariate/discrete/soliton",
    "univariate/continuous/skewnormal",
    "univariate/continuous/chi",
    "univariate/continuous/chisq",
    "univariate/continuous/erlang",
    "univariate/continuous/exponential",
    "univariate/continuous/gamma",
    "univariate/continuous/gumbel",
    "univariate/continuous/lindley",
    "univariate/continuous/logistic",
    "univariate/continuous/johnsonsu",
    "univariate/continuous/noncentralchisq",
    "univariate/continuous/weibull",
    "pdfnorm",
    "univariate/continuous/pareto",
    "univariate/continuous/rician",
    "functionals",
    "density_interface",
    "reshaped",
    "univariate/continuous/skewedexponentialpower",
    "univariate/discrete/discreteuniform",
    "univariate/continuous/tdist",
    "univariate/orderstatistic",
    "multivariate/jointorderstatistics",
    "multivariate/product",
    "eachvariate",
    "univariate/continuous/triangular",
    "statsapi",
    "univariate/continuous/inversegaussian",

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
    # "univariate/continuous/cosine",
    # "univariate/continuous/epanechnikov",
    # "univariate/continuous/fdist",
    # "univariate/continuous/frechet",
    # "univariate/continuous/generalizedextremevalue",
    # "univariate/continuous/generalizedpareto",
    # "univariate/continuous/inversegamma",
    # "univariate/continuous/ksdist",
    # "univariate/continuous/ksonesided",
    # "univariate/continuous/levy",
    # "univariate/continuous/noncentralbeta",
    # "univariate/continuous/noncentralf",
    # "univariate/continuous/normalinversegaussian",
    # "univariate/continuous/rayleigh",
    # "univariate/continuous/studentizedrange",
    # "univariate/continuous/symtriangular",
    # "univariate/continuous/tdist",
    # "univariate/continuous/triweight",
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

# to reduce redundancy, we might break this file down into separate `$t * "_utils.jl"` files
include("testutils.jl")

@testset "Distributions" begin
    @testset "Test $t" for t in tests
        include("$t.jl")
    end
end
