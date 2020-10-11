# Tests for Uniform ball distribution

using Distributions
using Test

function test_uniformball(n::Int)
    d = UniformBall(n)
    @test length(d) == n
    @test mean(d) == zeros(length(d))
    @test diag(cov(d)) == var(d)
    @test d == typeof(d)(params(d)...)
    @test d == deepcopy(d)
    @test partype(d) == Float64

    # conversions
    @test typeof(convert(UniformBall{Float32}, d)) == UniformBall{Float32}

    # Support
    x = normalize(rand(length(d)))

    if length(d) > 0
        @test !insupport(d, 100*x)
        @test pdf(d, 100*x) == 0.0
    else
        @test insupport(d, 100*x)
        @test pdf(d, 100*x) == 1.0
    end

    @test insupport(d, x)
    @test pdf(d, x) != 0.0

    @test insupport(d, 0.1*x)
    @test pdf(d, 0.1*x) != 0.0

    # Sampling
    X = [rand(d) for _ in 1:100_000]
    @test isapprox(mean(d), mean(X), atol=0.01)
    @test isapprox(var(d), var(X), atol=0.01)
    @test isapprox(cov(d), cov(X), atol=0.01)
end


## General testing

@testset "Testing UniformBall at $n" for n in 0:10
    test_uniformball(n)
end
