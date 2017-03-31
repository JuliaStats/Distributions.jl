# Tests for GeneralizedInverseGaussian

using Distributions
using Base.Test

srand(123)

# test constructors
d = GeneralizedInverseGaussian(1, 1, 1)
d2 = GeneralizedInverseGaussian(3.0, 2, -0.5)
@test typeof(d) == typeof(d2)

# test Sampling
g = Gamma(1, 1)
d = GeneralizedInverseGaussian(rand(g), rand(g), randn())
x = rand(d)
@test isa(x, Float64)
@test insupport(d, x)

n = 50
a = rand(g, n)
b = rand(g, n)
p = randn(n)
for i in eachindex(a, b, p)
    d = GeneralizedInverseGaussian(a[i], b[i], p[i])
    x = rand(d, 10_000)
    @test isa(x, Vector{Float64})
    @test all(insupport(d, x))
    @test isapprox(mean(d) / mean(x), 1, atol=.05)
end

# test evaluation
for i in eachindex(a, b, p)
    d = GeneralizedInverseGaussian(a[i], b[i], p[i])
    x = rand(d, 10_000)
    @test isapprox(cdf(d, mean(d)), mean(x .< mean(d)), atol=.05)
    @test isapprox(cdf(d, 0.1), mean(x .< 0.1), atol=.05)
    @test isapprox(cdf(d, 10), mean(x .< 10), atol=.05)
end

