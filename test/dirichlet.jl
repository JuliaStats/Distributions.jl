# Tests for Dirichlet distribution

using  Distributions
using Test, Random, LinearAlgebra


Random.seed!(34567)

d = Dirichlet(3, 2.0)

@test length(d) == 3
@test d.alpha == [2.0, 2.0, 2.0]
@test d.alpha0 == 6.0

@test mean(d) ≈ fill(1.0/3, 3)
@test cov(d)  ≈ [8 -4 -4; -4 8 -4; -4 -4 8] / (36 * 7)
@test var(d)  ≈ diag(cov(d))

@test pdf(d, [0.2, 0.3, 0.5])    ≈ 3.6
@test pdf(d, [0.4, 0.5, 0.1])    ≈ 2.4
@test logpdf(d, [0.2, 0.3, 0.5]) ≈ log(3.6)
@test logpdf(d, [0.4, 0.5, 0.1]) ≈ log(2.4)

x = rand(d, 100)
p = pdf(d, x)
lp = logpdf(d, x)
for i in 1 : size(x, 2)
    @test lp[i] ≈ logpdf(d, x[:,i])
    @test p[i]  ≈ pdf(d, x[:,i])
end

v = [2.0, 1.0, 3.0]
d = Dirichlet(v)

@test Dirichlet([2, 1, 3]).alpha == d.alpha

@test length(d) == length(v)
@test d.alpha == v
@test d.alpha0 == sum(v)
@test d == typeof(d)(params(d)...)

@test mean(d) ≈ v / sum(v)
@test cov(d)  ≈ [8 -2 -6; -2 5 -3; -6 -3 9] / (36 * 7)
@test var(d)  ≈ diag(cov(d))

@test pdf(d, [0.2, 0.3, 0.5])    ≈ 3.0
@test pdf(d, [0.4, 0.5, 0.1])    ≈ 0.24
@test logpdf(d, [0.2, 0.3, 0.5]) ≈ log(3.0)
@test logpdf(d, [0.4, 0.5, 0.1]) ≈ log(0.24)

x = rand(d, 100)
p = pdf(d, x)
lp = logpdf(d, x)
for i in 1 : size(x, 2)
    @test p[i]  ≈ pdf(d, x[:,i])
    @test lp[i] ≈ logpdf(d, x[:,i])
end

# Sampling

x = rand(d)
@test isa(x, Vector{Float64})
@test length(x) == 3

x = rand(d, 10)
@test isa(x, Matrix{Float64})
@test size(x) == (3, 10)


# Test MLE

n = 10000
x = rand(d, n)
x = x ./ sum(x, dims=1)

r = fit_mle(Dirichlet, x)
@test isapprox(r.alpha, d.alpha, atol=0.25)

# r = fit_mle(Dirichlet, x, fill(2.0, n))
# @test isapprox(r.alpha, d.alpha, atol=0.25)
