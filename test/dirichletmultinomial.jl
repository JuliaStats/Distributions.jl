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
@test !insupport(d, rand(5))
@test insupport(d, [2, 2, 2, 2, 2])
@test insupport(d, 2.0 * ones(5))
@test !insupport(d, 3.0 * ones(5))

for x in (2 * ones(5), [1, 2, 3, 4, 0], [3.0, 0.0, 3.0, 0.0, 4.0])
    @test_approx_eq(
        pdf(d, x),
        @compat factorial(d.n) * gamma(d.α0) / gamma(d.n + d.α0) * prod(gamma(d.α + x) ./ factorial.(x) ./ gamma(d.α))
    )
    @test_approx_eq(
        logpdf(d, x),
        @compat log(factorial(d.n)) + lgamma(d.α0) - lgamma(d.n + d.α0) + sum(lgamma(d.α + x) - log(factorial.(x)) - lgamma(d.α))
    )
end

# test Sampling
x = rand(d)
@test sum(x) == d.n
