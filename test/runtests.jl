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

using Distributions: continuous_distributions,
                     discrete_distributions,
                     matrix_distributions,
                     multivariate_distributions,
                     truncated_distributions

untested_distributions = [
    "beta",
    "betabinomial", # present but errors
    "betaprime",
    "biweight",
    "chisq",
    "cosine",
    "epanechnikov",
    "erlang",
    "exponential",
    "fdist",
    "frechet",
    "gamma",
    "generalizedextremevalue",
    "generalizedpareto",
    "geometric",
    "hypergeometric",
    "inversegamma",
    "inversegaussian",
    "ksdist",
    "ksonesided",
    "levy",
    "logistic",
    "noncentralbeta",
    "noncentralchisq",
    "noncentralf",
    "noncentralhypergeometric",
    "normalcanon",
    "normalinversegaussian",
    "pareto",
    "poisson",
    "rayleigh",
    "skellam",
    "studentizedrange",
    "symtriangular",
    "triangular",
    "triweight",
    "weibull",

    "inversewishart",
    "lkj",
    "matrixbeta",
    "matrixfdist", 
    "matrixnormal",
    "matrixtdist",
    "wishart",

    "mvnormalcanon",
]

const generic_tests = [
    "censored",
    # "common", # missing file
    "conversion",
    "convolution",
    "density_interface",
    # "eachvariate", # missing file
    "edgeworth",
    "fit", # extra file where there is none in /src
    "functionals",
    # "genericfit", # missing file
    # "genericrand", # missing file
    "gradlogpdf", # extra file where there is none in /src
    "pdfnorm",
    "qq",
    # "quantilealgs", # missing file
    "quantile_newton", # extra file where there is none in /src
    "reshaped",
    "samplers",
    # "show", # missing file
    "types", # extra file where there is none in /src
    "univariate_bounds", # extra file where there is none in /src
    "utils"
]

printstyled("Running tests:\n", color=:blue)

Random.seed!(345679)

# to reduce redundancy, we might break this file down into seperate `$t * "_utils.jl"` files
include("testutils.jl")

@testset "Distributions" begin
    for t in generic_tests
        @testset "Test $t" begin
            include("$t.jl")
        end
    end
    @testset "Test univariates" begin
        include("univariates.jl")
        @testset "Test locationscale" begin
            include(joinpath("univariate", "locationscale.jl"))
        end
        for dname in setdiff(discrete_distributions, untested_distributions)
            @testset "Test $dname" begin
                include(joinpath("univariate", "discrete", "$(dname).jl"))
            end
        end
        include("continuous.jl") # extra file where there is none in /src
        for dname in setdiff(continuous_distributions, untested_distributions)
            @testset "Test $dname" begin
                include(joinpath("univariate", "continuous", "$(dname).jl"))
            end
        end
    end
    @testset "Test multivariates" begin
        # include("multivariates.jl") # file missing
        for dname in setdiff(multivariate_distributions, untested_distributions)
            @testset "Test $dname" begin
                include(joinpath("multivariate", "$(dname).jl"))
            end
        end
    end
    @testset "Test matrixvariates" begin
        include("matrixreshaped.jl") # extra file where there is none in /src
        include("matrixvariates.jl")
        for dname in setdiff(matrix_distributions, untested_distributions)
            @testset "Test $dname" begin
                include(joinpath("matrix", "$(dname).jl"))
            end
        end
    end
    @testset "Test truncated" begin
        include("truncate.jl")
        for dname in setdiff(truncated_distributions, ["normal", "loguniform"])
            @testset "Test $dname" begin
                include(joinpath("truncated", "$(dname).jl"))
            end
        end
    end
end

# print method ambiguities
println("Potentially stale exports: ")
display(Test.detect_ambiguities(Distributions))
println()
