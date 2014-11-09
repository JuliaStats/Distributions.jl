using Distributions
using Base.Test

# Test for gradlogpdf on univariate distributions

@test_approx_eq_eps gradlogpdf(Beta(1.5, 3.0), 0.7) -5.9523809523809526 1.0e-8
@test_approx_eq_eps gradlogpdf(Chi(5.0), 5.5) -4.7727272727272725 1.0e-8
@test_approx_eq_eps gradlogpdf(Chisq(7.0), 12.0) -0.29166666666666663 1.0e-8
@test_approx_eq_eps gradlogpdf(Exponential(2.0), 7.0) -0.5 1.0e-8
@test_approx_eq_eps gradlogpdf(Gamma(9.0, 0.5), 11.0) -1.2727272727272727 1.0e-8
@test_approx_eq_eps gradlogpdf(Gumbel(3.5, 1.0), 4.0) -1.6065306597126334 1.0e-8
@test_approx_eq_eps gradlogpdf(Laplace(7.0), 34.0) -1.0 1.0e-8
@test_approx_eq_eps gradlogpdf(Logistic(-6.0), 1.0) -0.9981778976111987 1.0e-8
@test_approx_eq_eps gradlogpdf(LogNormal(5.5), 2.0) 1.9034264097200273 1.0e-8
@test_approx_eq_eps gradlogpdf(Normal(-4.5, 2.0), 1.6) -1.525 1.0e-8
@test_approx_eq_eps gradlogpdf(TDist(8.0), 9.1) -0.9018830525272548 1.0e-8
@test_approx_eq_eps gradlogpdf(Weibull(2.0), 3.5) -6.714285714285714 1.0e-8

# Test for gradlogpdf on multivariate distributions

@test_approx_eq_eps gradlogpdf(MvNormal([1., 2.], [1. 0.1; 0.1 1.]), [0.7, 0.9]) [0.191919191919192, 1.080808080808081] 1.0e-8
@test_approx_eq_eps gradlogpdf(MvTDist(5., [1., 2.], [1. 0.1; 0.1 1.]), [0.7, 0.9]) [0.2150711513583442, 1.2111901681759383] 1.0e-8
