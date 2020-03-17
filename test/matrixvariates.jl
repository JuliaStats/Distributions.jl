using Distributions
using Random
using LinearAlgebra
using PDMats
using Statistics
using HypothesisTests
using Test
import Distributions: _univariate, _rand_params

#  =============================================================================
#  Test matrix-variate against the univariate it collapses to in the 1 x 1 case
#  =============================================================================

function test_against_univariate(D::MatrixDistribution, d::UnivariateDistribution)
    X = rand(D)
    x = X[1]
    α = 0.05
    M = 10000
    matvardraws = [rand(D)[1] for m in 1:M]
    @test logpdf(D, X) ≈ logpdf(d, x)
    @test mean(D)[1] ≈ mean(d)
    @test var(D)[1] ≈ var(d)
    @test pvalue(ExactOneSampleKSTest(matvardraws, d)) >= α
    nothing
end

function test_against_univariate(dist::Type)
    D = dist(_rand_params(dist, Float64, 1, 1)...)
    d = _univariate(D)
    test_against_univariate(D, d)
    nothing
end

#  =============================================================================
#  main method
#  =============================================================================

function test_matrixvariate(dist::Type)
    #  Baseline test
    test_against_univariate(dist)
    #  test_against_multivariate
    #  test_against_r
    #  distribution_specific shit
    nothing
end

#  =============================================================================
#  run unit tests for matrix-variate distributions
#  =============================================================================

matrixvariates = [MatrixNormal,
                  MatrixTDist,
                  Wishart,
                  InverseWishart,
                  MatrixBeta,
                  MatrixFDist]

for distribution in matrixvariates
    println("    testing $(distribution)")
    @testset "$(distribution)" begin
        test_matrixvariate(distribution)
    end
end
