# Testing continuous univariate distributions

using Distributions
using Base.Test

### load reference data
#
#   Note
#   -------
#   To generate the reference data:
#   (1) make sure that python, numpy, and scipy are installed in your system
#   (2) enter the sub-directory test
#   (3) run: python discrete_ref.py > discrete_ref.csv
#
#   For most cases, you don't have. You only need to run this when you
#   implement a new distribution and want to add new test cases, then
#   you should add the new test cases to discrete_ref.py and run this
#   procedure to update the reference data.
#
n_tsamples = 100


# additional distributions that have no direct counterparts in scipy
println("    -----")

for distr in [
    Biweight(),
    Biweight(1,3),
    Epanechnikov(),
    Epanechnikov(1,3),
    Frechet(0.5, 1.0),
    Frechet(3.0, 1.0),
    Frechet(20.0, 1.0),
    Frechet(120.0, 1.0),
    Frechet(0.5, 2.0),
    Frechet(3.0, 2.0),
    GeneralizedPareto(),
    GeneralizedPareto(1.0, 1.0),
    GeneralizedPareto(1.0, 1.0, 1.0),
    GeneralizedPareto(0.1, 2.0, 0.0),
    GeneralizedPareto(0.0, 0.5, 0.0),
    GeneralizedPareto(-1.5, 0.5, 2.0),
    InverseGaussian(1.0, 1.0),
    InverseGaussian(2.0, 7.0),
    Levy(0.0, 1.0),
    Levy(2.0, 8.0),
    Levy(3.0, 3.0),
    LogNormal(0.0, 1.0),
    LogNormal(0.0, 2.0),
    LogNormal(3.0, 0.5),
    LogNormal(3.0, 1.0),
    LogNormal(3.0, 2.0),
    NoncentralBeta(2,2,0),
    NoncentralBeta(2,6,5),
    NoncentralChisq(2,2),
    NoncentralChisq(2,5),
    NoncentralF(2,2,2),
    NoncentralF(8,10,5),
    NoncentralT(2,2),
    NoncentralT(10,2),
    Triweight(),
    Triweight(1,3),
]
    println("    testing $(distr)")
    test_distr(distr, n_tsamples)
end


# for distr in [
#     VonMises(0.0, 1.0),
#     VonMises(0.5, 1.0),
#     VonMises(0.5, 2.0) ]

#     println("    testing $(distr)")
#     test_samples(distr, n_tsamples)
# end

# Test for non-Float64 input
using ForwardDiff
@test string(logpdf(Normal(0,1),big(1))) == "-1.418938533204672741780329736405617639861397473637783412817151540482765695927251"
@test_approx_eq derivative(t -> logpdf(Normal(1.0, 0.15), t), 2.5) -66.66666666666667
@test derivative(t -> pdf(Normal(t, 1.0), 0.0), 0.0) == 0.0
