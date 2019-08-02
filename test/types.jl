# Test type relations

using Distributions
using ForwardDiff: Dual

@assert UnivariateDistribution <: Distribution
@assert MultivariateDistribution <: Distribution
@assert MatrixDistribution <: Distribution

@assert IntegerDistribution <: Distribution
@assert ContinuousDistribution <: Distribution

@assert IntegerUnivariateDistribution <: IntegerDistribution 
@assert IntegerUnivariateDistribution <: UnivariateDistribution
@assert ContinuousUnivariateDistribution <: ContinuousDistribution
@assert ContinuousUnivariateDistribution <: UnivariateDistribution
@assert IntegerMultivariateDistribution <: IntegerDistribution
@assert IntegerMultivariateDistribution <: MultivariateDistribution
@assert ContinuousMultivariateDistribution <: ContinuousDistribution
@assert ContinuousMultivariateDistribution <: MultivariateDistribution
@assert IntegerMatrixDistribution <: IntegerDistribution
@assert IntegerMatrixDistribution <: MatrixDistribution
@assert ContinuousMatrixDistribution <: ContinuousDistribution
@assert ContinuousMatrixDistribution <: MatrixDistribution

@testset "Test Sample Type" begin
    for T in (Float64,Float32,Dual{Nothing,Float64,0})
        @testset "Type $T" begin
            for d in (MvNormal,MvLogNormal,MvNormalCanon,Dirichlet)
                dist = d(map(T,ones(2)))
                @test eltype(dist) == T
                @test eltype(rand(dist)) == eltype(dist)
            end
            dist = Distributions.mvtdist(map(T,1.0),map(T,[1.0 0.0; 0.0 1.0]))
            @test eltype(dist) == T
            @test eltype(rand(dist)) == eltype(dist)
        end
    end
end
