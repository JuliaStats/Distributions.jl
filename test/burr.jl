# Tests for Burr

using Distributions
using Base.Test

srand(123)

# test constructors
d = Burr(1, 1, 1)
d2 = Burr(3.0, 2, 5.0)
@test typeof(d) == typeof(d2)

# test Sampling
g = Gamma(5, 2)
d = Burr(rand(g), rand(g), rand(g))
x = rand(d)
@test isa(x, Float64)
@test insupport(d, x)

n = 50
a = rand(g, n)
b = rand(g, n)
p = rand(g, n)
for i in eachindex(a, b, p)
    d = Burr(a[i], b[i], p[i])
    x = rand(d, 10_000)
    @test isa(x, Vector{Float64})
    @test all(insupport(d, x))
    @test isapprox(mean(d) / mean(x), 1, atol=.05)
end

# test evaluation
for i in eachindex(a, b, p)
    d = Burr(a[i], b[i], p[i])
    x = rand(d, 10_000)
    @test isapprox(cdf(d, mean(d)), mean(x .< mean(d)), atol=.05)
    @test isapprox(cdf(d, 0.1), mean(x .< 0.1), atol=.05)
    @test isapprox(cdf(d, 10), mean(x .< 10), atol=.05)
end