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
                     truncated_distributions

const matrix_distributions = [
    "wishart",
    "inversewishart",
    "matrixnormal",
    "matrixtdist",
    "matrixbeta",
    "matrixfdist", 
    "lkj"
]

const multivariate_distributions = [
    "dirichlet",
    "multinomial",
    "dirichletmultinomial",
    "mvnormal",
    "mvnormalcanon",
    "mvlognormal",
    "mvtdist",
    "product",
    "vonmisesfisher"
]

const generic_tests = [
    "censored",
    "common",
    "conversion",
    "convolution",
    "density_interface",
    "deprecates",
    "eachvariate",
    "edgeworth",
    "estimators",
    "functionals",
    "genericfit",
    "genericrand",
    "pdfnorm",
    "qq",
    "quantilealgs",
    "reshaped",
    "samplers",
    "show",
    "utils"
]

const tests_by_method_or_trait = [
    "fit",
    "gradlogpdf",
    "types",
    "univariate_bounds",
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
    for t in tests_by_method_or_trait
        @testset "Test $t" begin
            include(joinpath("_methods", "$t.jl"))
        end
    end
    @testset "Test univariates" begin
        include("univariates.jl")
        @testset "Test locationscale" begin
            include(joinpath("univariate", "locationscale.jl"))
        end
        for dname in discrete_distributions
            @testset "Test $dname" begin
                include(joinpath("univariate", "discrete", "$(dname).jl"))
            end
        end
        for dname in continuous_distributions
            @testset "Test $dname" begin
                include(joinpath("univariate", "continuous", "$(dname).jl"))
            end
        end
    end
    @testset "Test multivariates" begin
        include("multivariates.jl")
        for dname in multivariate_distributions
            @testset "Test $dname" begin
                include(joinpath("multivariate", "$(dname).jl"))
            end
        end
    end
    @testset "Test matrixvariates" begin
        include("matrixvariates.jl")
        for dname in matrix_distributions
            @testset "Test $dname" begin
                include(joinpath("matrix", "$(dname).jl"))
            end
        end
    end
    @testset "Test truncated" begin
        include("truncate.jl")
        for dname in truncated_distributions
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
