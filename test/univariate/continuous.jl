# Testing continuous univariate distributions

using Distributions, Random
using Test

using Calculus: derivative

n_tsamples = 100

# additional distributions that have no direct counterparts in R references
@testset "Testing $(distr)" for distr in [Biweight(),
                                          Biweight(1,3),
                                          Epanechnikov(),
                                          Epanechnikov(1,3),
                                          Triweight(),
                                          Triweight(2),
                                          Triweight(1, 3),
                                          Triweight(1)]

    test_distr(distr, n_tsamples; testquan=false)
end

# Test for non-Float64 input
using ForwardDiff
@test string(logpdf(Normal(0,1),big(1))) == "-1.418938533204672741780329736405617639861397473637783412817151540482765695927251"
@test derivative(t -> logpdf(Normal(1.0, 0.15), t), 2.5) ≈ -66.66666666666667
@test derivative(t -> pdf(Normal(t, 1.0), 0.0), 0.0) == 0.0


@testset "Normal distribution with non-standard (ie not Float64) parameter types" begin
    n32 = Normal(1f0, 0.1f0)
    n64 = Normal(1., 0.1)
    nbig = Normal(big(pi), big(ℯ))

    @test eltype(typeof(n32)) === Float32
    @test eltype(rand(n32)) === Float32
    @test eltype(rand(n32, 4)) === Float32

    @test eltype(typeof(n64)) === Float64
    @test eltype(rand(n64)) === Float64
    @test eltype(rand(n64, 4)) === Float64
end

# Test for numerical problems
@test pdf(Logistic(6,0.01),-2) == 0

@testset "Normal with std=0" begin
    d = Normal(0.5,0.0)
    @test pdf(d, 0.49) == 0.0
    @test pdf(d, 0.5) == Inf
    @test pdf(d, 0.51) == 0.0

    @test cdf(d, 0.49) == 0.0
    @test cdf(d, 0.5) == 1.0
    @test cdf(d, 0.51) == 1.0

    @test ccdf(d, 0.49) == 1.0
    @test ccdf(d, 0.5) == 0.0
    @test ccdf(d, 0.51) == 0.0

    @test quantile(d, 0.0) == -Inf
    @test quantile(d, 0.49) == 0.5
    @test quantile(d, 0.5) == 0.5
    @test quantile(d, 0.51) == 0.5
    @test quantile(d, 1.0) == +Inf

    @test rand(d) == 0.5
    @test rand(MersenneTwister(123), d) == 0.5
end

# Test for parameters beyond those supported in R references
@testset "VonMises with large kappa" begin
    d = VonMises(1.1, 1000)
    @test var(d) ≈ 0.0005001251251957198
    @test entropy(d) ≈ -2.034688918525470
    @test cf(d, 2.5) ≈ -0.921417 + 0.38047im atol=1e-6
    @test pdf(d, 0.5) ≈ 1.758235814051e-75
    @test logpdf(d, 0.5) ≈ -172.1295710466005
    @test cdf(d, 1.0) ≈ 0.000787319 atol=1e-9
end

@testset "NormalInverseGaussian random repeatable and basic metrics" begin
    rng = Random.MersenneTwister(42)
    rng2 = copy(rng)
    µ = 0.0
    α = 1.0
    β = 0.5
    δ = 3.0
    g = sqrt(α^2 - β^2)
    d = NormalInverseGaussian(μ, α, β, δ)
    v1 = rand(rng, d)
    v2 = rand(rng, d)
    v3 = rand(rng2, d)
    @test v1 ≈ v3
    @test v1 ≉ v2

    @test mean(d) ≈ µ + β * δ / g
    @test var(d) ≈ δ * α^2 / g^3
    @test skewness(d) ≈ 3β/(α*sqrt(δ*g))
end

@testset "edge cases" begin
    # issue #1371: cdf should not return -0.0
    @test cdf(Rayleigh(1), 0) === 0.0
    @test cdf(Rayleigh(1), -10) === 0.0
end
