# Tests for Burr

using Distributions
using Test
using Random

Random.seed!(1234)

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

# test parameter
d = Burr()
@test params(d) == (1, 1, 1)
d = Burr(6.0, 4.5)
@test params(d) == (6.0, 4.5, 1)
d = Burr(2, 3, 5.0)
@test params(d) == (2, 3, 5.0)
# d = Burr(1.5, 6.3, 4.2)
# @test scale(d) == 4.20

# Non-extreme case: mean, variance, median
n = 50
a = rand(g, n) .+ 1.5
b = rand(g, n) .+ 1.5
p = rand(g, n)
for i in eachindex(a, b, p)
    d = Burr(a[i], b[i], p[i])
    x = rand(d, 100_000)
    @test isa(x, Vector{Float64})
    @test all(insupport(d, x))
    @test isapprox(mean(d) / mean(x), 1, atol=.05)
    @test isapprox(var(d) / var(x), 1, atol=.05)
    @test isapprox(median(d) / median(x), 1, atol=.05)
    @test isapprox(quantile(d, 0.05) / quantile(x, 0.05), 1, atol=.05)
    @test isapprox(quantile(d, 0.10) / quantile(x, 0.10), 1, atol=.05)
    @test isapprox(quantile(d, 0.30) / quantile(x, 0.30), 1, atol=.05)
    @test isapprox(quantile(d, 0.70) / quantile(x, 0.70), 1, atol=.05)
    @test isapprox(quantile(d, 0.90) / quantile(x, 0.90), 1, atol=.05)
    @test isapprox(quantile(d, 0.95) / quantile(x, 0.95), 1, atol=.05)
end

# Extreme case: mean = Inf
a = [0.1, 2.0, 1.0, 0.75]
b = [2.0, 0.3, 0.8, 1.01]
p = [5.0, 4.0, 2.5, 3.99]
for i in eachindex(a, b, p)
    d = Burr(a[i], b[i], p[i])
    @test isinf(mean(d))
end

# Extreme case: mean < Inf, var = Inf
a = [0.6, 2.0, 1.5, 0.99]
b = [2.0, 0.7, 0.8, 1.10]
p = [5.0, 4.0, 2.5, 3.99]
for i in eachindex(a, b, p)
    d = Burr(a[i], b[i], p[i])
    @test !isinf(mean(d))
    @test isinf(var(d))
end


# Empirical: cdf, ccdf
n = 50
a = rand(g, n)
b = rand(g, n)
p = rand(g, n)
for i in eachindex(a, b, p)
    d = Burr(a[i], b[i], p[i])
    x = rand(d, 10_000)
    @test isapprox(cdf(d, mean(d)), mean(x .< mean(d)), atol=.05)
    @test isapprox(cdf(d, median(d)), mean(x .< median(d)), atol=.05)
    @test isapprox(cdf(d, 0.01), mean(x .< 0.01), atol=.05)
    @test isapprox(cdf(d, 0.1), mean(x .< 0.1), atol=.05)
    @test isapprox(cdf(d, 10), mean(x .< 10), atol=.05)
    @test isapprox(cdf(d, 100), mean(x .< 100), atol=.05)
    @test isapprox(cdf(d, 1000), mean(x .< 1000), atol=.05)
    @test isapprox(ccdf(d, 0.01), mean(x .> 0.01), atol=.05)
    @test isapprox(ccdf(d, 0.1), mean(x .> 0.1), atol=.05)
    @test isapprox(ccdf(d, 10), mean(x .> 10), atol=.05)
    @test isapprox(ccdf(d, 100), mean(x .> 100), atol=.05)
    @test isapprox(ccdf(d, 1000), mean(x .> 1000), atol=.05)
end

# pdf & logpdf, cdf & logcdf
n = 50
a = rand(g, n)
b = rand(g, n)
p = rand(g, n)
for i in eachindex(a, b, p)
    d = Burr(a[i], b[i], p[i])
    x = rand(d, 500)
    xx = -20:5:2000
    ii = [-Inf Inf]
    @test isapprox(pdf.(d, x), exp.(logpdf.(d, x)), atol=.05)
    @test isapprox(pdf.(d, xx), exp.(logpdf.(d, xx)), atol=.05)
    @test isapprox(pdf.(d, ii), exp.(logpdf.(d, ii)), atol=.05)
    @test isapprox(cdf.(d, x), exp.(logcdf.(d, x)), atol=.05)
    @test isapprox(cdf.(d, xx), exp.(logcdf.(d, xx)), atol=.05)
    @test isapprox(cdf.(d, ii), exp.(logcdf.(d, ii)), atol=.05)
    @test isapprox(ccdf.(d, x), exp.(logccdf.(d, x)), atol=.05)
    @test isapprox(ccdf.(d, xx), exp.(logccdf.(d, xx)), atol=.05)
    @test isapprox(ccdf.(d, ii), exp.(logccdf.(d, ii)), atol=.05)
end