# Tests for DirichletMultinomial


using Distributions
using Test, Random, SpecialFunctions

import SpecialFunctions: factorial

Random.seed!(123)


# Test Constructors
d = DirichletMultinomial(10, ones(5))
d2 = DirichletMultinomial(10, ones(Int, 5))
@test typeof(d) == typeof(d2)

α = rand(5)
d = DirichletMultinomial(10, α)

# test parameters
@test length(d) == 5
@test ntrials(d) == 10
@test params(d) == (10, α)
@test d.α == α
@test d.α0 == sum(α)
@test partype(d) == eltype(α)

# test statistics
@test mean(d) ≈ α * (d.n / d.α0)
p = d.α / d.α0
@test var(d)  ≈ d.n * (d.n + d.α0) / (1 + d.α0) .* p .* (1.0 .- p)
x = rand(d, 10_000)

# test statistics with mle fit
d = fit(DirichletMultinomial, x)
@test isapprox(mean(d), vec(mean(x, dims=2)), atol=.5)
@test isapprox(var(d) , vec(var(x, dims=2)) , atol=.5)
@test isapprox(cov(d) , cov(x, dims=2)      , atol=.5)

# test Evaluation
d = DirichletMultinomial(10, 5)
@test typeof(d) == DirichletMultinomial{Float64}
@test !insupport(d, rand(5))
@test insupport(d, [2, 2, 2, 2, 2])
@test insupport(d, 2.0 * ones(5))
@test !insupport(d, 3.0 * ones(5))

for x in (2 * ones(5), [1, 2, 3, 4, 0], [3.0, 0.0, 3.0, 0.0, 4.0], [0, 0, 0, 0, 10])
    local x
    @test pdf(d, x) ≈
        factorial(d.n) * gamma(d.α0) / gamma(d.n + d.α0) * prod(gamma.(d.α + x) ./ factorial.(x) ./ gamma.(d.α))
    @test logpdf(d, x) ≈
        log(factorial(d.n)) + lgamma(d.α0) - lgamma(d.n + d.α0) + sum(lgamma.(d.α + x) - log.(factorial.(x)) - lgamma.(d.α))
end

# test Sampling
x = rand(d)
@test isa(x, Vector{Int})
@test sum(x) == d.n
@test length(x) == length(d)
@test insupport(d, x)

x = rand(d, 50)
@test all(x -> (x >= 0), x)
@test size(x, 1) == length(d)
@test size(x, 2) == 50
@test all(sum(x, dims=1) .== ntrials(d))
@test all(insupport(d, x))

# test MLE
x = rand(d, 10_000)
ss = suffstats(DirichletMultinomial, x)
@test size(ss.s, 1) == length(d)
@test size(ss.s, 2) == ntrials(d)
mle = fit(DirichletMultinomial, x)
@test isapprox(mle.α, d.α, atol=.2)

# test MLE with weights
for w in (.1 * ones(10_000), ones(10_000), 10 * ones(10_000))
    mle2 = fit(DirichletMultinomial, x, w)
    @test mle.α ≈ mle2.α
end
