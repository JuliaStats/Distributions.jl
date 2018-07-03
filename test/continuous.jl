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
