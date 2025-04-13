# Test type relations

using Distributions
using ForwardDiff: Dual

@assert UnivariateDistribution <: Distribution
@assert MultivariateDistribution <: Distribution
@assert MatrixDistribution <: Distribution

@assert DiscreteDistribution <: Distribution
@assert ContinuousDistribution <: Distribution

@assert DiscreteUnivariateDistribution <: DiscreteDistribution
@assert DiscreteUnivariateDistribution <: UnivariateDistribution
@assert ContinuousUnivariateDistribution <: ContinuousDistribution
@assert ContinuousUnivariateDistribution <: UnivariateDistribution
@assert DiscreteMultivariateDistribution <: DiscreteDistribution
@assert DiscreteMultivariateDistribution <: MultivariateDistribution
@assert ContinuousMultivariateDistribution <: ContinuousDistribution
@assert ContinuousMultivariateDistribution <: MultivariateDistribution
@assert DiscreteMatrixDistribution <: DiscreteDistribution
@assert DiscreteMatrixDistribution <: MatrixDistribution
@assert ContinuousMatrixDistribution <: ContinuousDistribution
@assert ContinuousMatrixDistribution <: MatrixDistribution

@testset "Test Sample Type" begin
    for T in (Float64,Float32,Dual{Nothing,Float64,0})
        @testset "Type $T" begin
            dists = (
                MvNormal(Diagonal(ones(T, 2))),
                MvLogNormal(Diagonal(ones(T, 2))),
                MvNormalCanon(Diagonal(ones(T, 2))),
                Dirichlet(ones(T, 2)),
                Distributions.mvtdist(one(T), Matrix{T}(I, 2, 2)),
            )
            for dist in dists
                @test eltype(typeof(dist)) === T
                @test eltype(rand(dist)) === eltype(dist)
            end
        end
    end
end

@testset "equality" begin
    dist1 = Normal(1, 1)
    dist2 = Normal(1.0, 1.0)

    # Check h is used
    @test hash(dist1, UInt(1)) != hash(dist1, UInt(2))

    @test dist1 == deepcopy(dist1)
    @test hash(dist1) == hash(deepcopy(dist1))
    @test dist1 == dist2
    @test isequal(dist1, dist2)
    @test isapprox(dist1, dist2)
    @test hash(dist1) == hash(dist2)

    dist3 = Normal(1, 0.8)
    @test dist1 != dist3
    @test !isequal(dist1, dist3)
    @test !isapprox(dist1, dist3)
    @test hash(dist1) != hash(dist3)
    @test isapprox(dist1, dist3, atol=0.3)

    dist4 = LogNormal(1, 1)
    @test dist1 != dist4
    @test !isequal(dist1, dist4)
    @test !isapprox(dist1, dist4)
    @test hash(dist1) != hash(dist4)
end
