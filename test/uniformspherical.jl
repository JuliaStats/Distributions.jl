# Tests for Uniform Spherical distribution

using Distributions
using Test

function test_uniformspherical(n::Int)
    d = UniformSpherical(n)
    @test length(d) == n+1
    @test mean(d) == zeros(length(d))
    @test iszero(concentration(d))
    @test isdiag(cov(d))
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
    X = rand(d, 100_000)
    @test isapprox(mean(d), mean(X; dims=2), atol=0.01)
    @test isapprox(var(d), var(X; dims=2), atol=0.01)
    @test isapprox(cov(d), cov(X; dims=2), atol=0.01)

    # "Fitting"
    X = randn(n+1, 100)
    U = fit_mle(UniformSpherical, X)
    @test length(U) == size(X, 1)
end


## General testing

@testset "Entropy/Normalization for UniformSpherical" begin
    @test exp(entropy(UniformSpherical(0))) ≈ 2
    @test exp(entropy(UniformSpherical(1))) ≈ 2π
    @test exp(entropy(UniformSpherical(2))) ≈ 4π
end

@testset "Testing UniformSpherical at $n" for n in 0:10
    test_uniformspherical(n)
end
