# Tests for Uniform Spherical distribution

using Distributions
using Test

function test_uniformspherical(n::Int)
    d = UniformSpherical(n)
    @test length(d) == n+1
    @test mean(d) == zeros(length(d))
    @test diag(cov(d)) == var(d)
    @test d == typeof(d)(params(d)...)
    @test d == deepcopy(d)
    @test partype(d) == Float64

    # conversions
    @test typeof(convert(UniformSpherical{Float32}, d)) == UniformSpherical{Float32}

    # Support
    x = normalize(rand(length(d)))

    @test !insupport(d, 100*x)
    @test pdf(d, 100*x) == 0.0

    @test insupport(d, x)
    @test pdf(d, x) != 0.0

    @test !insupport(d, 0.1*x)
    @test pdf(d, 0.1*x) == 0.0

    # Sampling
    X = [rand(d) for _ in 1:100_000]
    @test isapprox(mean(d), mean(X), atol=0.01)
    @test isapprox(var(d), var(X), atol=0.01)
    @test isapprox(cov(d), cov(X), atol=0.01)
end


## General testing

@testset "Testing UniformSpherical at $n" for n in 0:10
    test_uniformspherical(n)
end
