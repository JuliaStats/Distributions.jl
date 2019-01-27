# Testing continuous univariate distributions

using Distributions
using Test

using Calculus: derivative

n_tsamples = 100

# additional distributions that have no direct counterparts in R references
for distr in [
    Biweight(),
    Biweight(1,3),
    Epanechnikov(),
    Epanechnikov(1,3),
    Triweight(),
    Triweight(2),
    Triweight(1, 3),
    Triweight(1),
]
    println("    testing $(distr)")
    test_distr(distr, n_tsamples; testquan=false)
end

# Test for non-Float64 input
using ForwardDiff
@test string(logpdf(Normal(0,1),big(1))) == "-1.418938533204672741780329736405617639861397473637783412817151540482765695927251"
@test derivative(t -> logpdf(Normal(1.0, 0.15), t), 2.5) ≈ -66.66666666666667
@test derivative(t -> pdf(Normal(t, 1.0), 0.0), 0.0) == 0.0

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
end    
