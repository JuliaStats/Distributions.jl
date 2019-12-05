# Test type relations

using Distributions
using ForwardDiff: Dual
using Test

@test UnivariateDistribution <: Distribution
@test MultivariateDistribution <: Distribution
@test MatrixDistribution <: Distribution

@test DiscreteDistribution <: Distribution
@test ContinuousDistribution <: Distribution

@test DiscreteUnivariateDistribution <: DiscreteDistribution 
@test DiscreteUnivariateDistribution <: UnivariateDistribution
@test ContinuousUnivariateDistribution <: ContinuousDistribution
@test ContinuousUnivariateDistribution <: UnivariateDistribution
@test DiscreteMultivariateDistribution <: DiscreteDistribution
@test DiscreteMultivariateDistribution <: MultivariateDistribution
@test ContinuousMultivariateDistribution <: ContinuousDistribution
@test ContinuousMultivariateDistribution <: MultivariateDistribution
@test DiscreteMatrixDistribution <: DiscreteDistribution
@test DiscreteMatrixDistribution <: MatrixDistribution
@test ContinuousMatrixDistribution <: ContinuousDistribution
@test ContinuousMatrixDistribution <: MatrixDistribution

@test_skip ValueSupport
@test_skip Discrete
@test_skip Continuous

@test_skip DiscreteUnivariateDistribution
@test_skip ContinuousUnivariateDistribution
@test_skip DiscreteMultivariateDistribution
@test_skip ContinuousMultivariateDistribution
@test_skip DiscreteMatrixDistribution
@test_skip ContinuousMatrixDistribution

@testset "Test Sample Type" begin
    for T in (Float64,Float32,Dual{Nothing,Float64,0})
        @testset "Type $T" begin
            for d in (MvNormal,MvLogNormal,MvNormalCanon,Dirichlet)
                dist = d(map(T,ones(2)))
                @test eltype(typeof(dist)) == T
                @test eltype(rand(dist)) == eltype(dist)
            end
            dist = Distributions.mvtdist(map(T,1.0),map(T,[1.0 0.0; 0.0 1.0]))
            @test eltype(typeof(dist)) == T
            @test eltype(rand(dist)) == eltype(dist)
        end
    end
end
