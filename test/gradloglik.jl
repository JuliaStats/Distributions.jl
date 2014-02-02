using Distributions
using Base.Test

# Test for gradloglik on univariate distributions

@test_approx_eq_eps gradloglik(Beta(1.5, 3.0), 0.7) -5.9523809523809526 1.0e-8
@test_approx_eq_eps gradloglik(Chi(5.0), 5.5) -4.7727272727272725 1.0e-8
@test_approx_eq_eps gradloglik(Chisq(7.0), 12.0) -0.29166666666666663 1.0e-8
@test_approx_eq_eps gradloglik(Exponential(2.0), 7.0) -0.5 1.0e-8
@test_approx_eq_eps gradloglik(Gamma(9.0, 0.5), 11.0) -1.2727272727272727 1.0e-8
@test_approx_eq_eps gradloglik(Gumbel(3.5, 1.0), 4.0) -1.6065306597126334 1.0e-8
@test_approx_eq_eps gradloglik(Laplace(7.0), 34.0) -1.0 1.0e-8
@test_approx_eq_eps gradloglik(Logistic(-6.0), 1.0) -0.9981778976111987 1.0e-8
@test_approx_eq_eps gradloglik(LogNormal(5.5), 2.0) 1.9034264097200273 1.0e-8
@test_approx_eq_eps gradloglik(Normal(-4.5, 2.0), 1.6) -1.525 1.0e-8
@test_approx_eq_eps gradloglik(TDist(8.0), 9.1) -0.9018830525272548 1.0e-8
@test_approx_eq_eps gradloglik(Weibull(2.0), 3.5) -6.714285714285714 1.0e-8

# Test for gradloglik on multivariate distributions

@test_approx_eq_eps gradloglik(MvNormal([1., 2.], [1. 0.1; 0.1 1.]), [0.7, 0.9]) [0.191919191919192, 1.080808080808081] 1.0e-8
@test_approx_eq_eps gradloglik(MvTDist(5., [1., 2.], [1. 0.1; 0.1 1.]), [0.7, 0.9]) [0.2150711513583442, 1.2111901681759383] 1.0e-8
