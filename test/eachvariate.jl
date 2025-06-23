using Distributions
using ChainRulesTestUtils
using ChainRulesTestUtils: FiniteDifferences

using Random
using Test

# Without this, `to_vec` will also include the `axes` field of `EachVariate`.
function FiniteDifferences.to_vec(xs::Distributions.EachVariate{V}) where {V}
    vals, vals_from_vec = FiniteDifferences.to_vec(xs.parent)
    return vals, x -> Distributions.EachVariate{V}(vals_from_vec(x))
end

# MWE in #1817
struct FooEachvariate <: Sampleable{Multivariate, Continuous} end
Base.length(::FooEachvariate) = 3
Base.eltype(::FooEachvariate) = Float64
function Distributions._rand!(rng::AbstractRNG, ::FooEachvariate, x::AbstractVector{<:Real})
    return rand!(rng, x)
end

@testset "eachvariate.jl" begin
    @testset "ChainRules" begin
        xs = randn(2, 3, 4, 5)
        test_rrule(Distributions.EachVariate{1}, xs)
        test_rrule(Distributions.EachVariate{2}, xs)
        test_rrule(Distributions.EachVariate{3}, xs)
    end

    @testset "No variates (#1817)" begin
        @test size(Distributions.eachvariate(rand(0), Univariate)) == (0,)
        @test size(Distributions.eachvariate(rand(3, 0, 1), Univariate)) == (3, 0, 1)
        @test size(Distributions.eachvariate(rand(3, 2, 0), Univariate)) == (3, 2, 0)

        @test size(Distributions.eachvariate(rand(4, 0), Multivariate)) == (0,)
        @test size(Distributions.eachvariate(rand(4, 0, 2), Multivariate)) == (0, 2)
        @test size(Distributions.eachvariate(rand(4, 5, 0), Multivariate)) == (5, 0)
        @test size(Distributions.eachvariate(rand(4, 5, 0, 2), Multivariate)) == (5, 0, 2)

        draws = @inferred(rand(FooEachvariate(), 0))
        @test draws isa Matrix{Float64}
        @test size(draws) == (3, 0)
    end
end
