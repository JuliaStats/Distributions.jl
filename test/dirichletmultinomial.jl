# Tests for DirichletMultinomial


using Distributions
using Base.Test

srand(123)

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
@test_approx_eq(mean(d), α * (d.n / d.α0))
p = d.α / d.α0
@test_approx_eq(var(d), d.n * (d.n + d.α0) / (1 + d.α0) .* p .* (1.0 - p))

# test Evaluation
d = DirichletMultinomial(10, 5)
@test typeof(d) == DirichletMultinomial{Float64}
@test !insupport(d, rand(5))
@test insupport(d, [2, 2, 2, 2, 2])
@test insupport(d, 2.0 * ones(5))
@test !insupport(d, 3.0 * ones(5))

for x in (2 * ones(5), [1, 2, 3, 4, 0], [3.0, 0.0, 3.0, 0.0, 4.0], [0, 0, 0, 0, 10])
    @test_approx_eq(
        pdf(d, x),
        @compat factorial(d.n) * gamma(d.α0) / gamma(d.n + d.α0) * prod(gamma(d.α + x) ./ factorial.(x) ./ gamma(d.α))
    )
    @test_approx_eq(
        logpdf(d, x),
        @compat log(factorial(d.n)) + lgamma(d.α0) - lgamma(d.n + d.α0) + sum(lgamma(d.α + x) - log.(factorial.(x)) - lgamma(d.α))
    )
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
@test all(sum(x, 1) .== ntrials(d))
@test all(insupport(d, x))


# test MLE
x = rand(d, 10_000)
ss = suffstats(DirichletMultinomial, x)
@test size(ss.s, 1) == length(d)
@test size(ss.s, 2) == ntrials(d)
mle = fit(DirichletMultinomial, x)
@test_approx_eq_eps mle.α d.α .2

# test MLE with weights
ss2 = suffstats(DirichletMultinomial, x, ones(10_000))
@test ss2.s == ss.s
mle2 = fit(DirichletMultinomial, x, ones(10_000))
@test mle.α == mle2.α
