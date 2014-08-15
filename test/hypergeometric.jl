using Distributions
using Base.Test

# simple example
ns = 4
nf = 6
n = 3

d = Hypergeometric(ns, nf, n)

@test_approx_eq mean(d) 6.0/5.0
@test_approx_eq var(d) 14.0/25.0
@test_approx_eq skewness(d) 1.0/(2.0*sqrt(14.0))
@test_approx_eq kurtosis(d) 128.0/49.0-3
@test mode(d) == 1

@test quantile(d, 0.05) == 0
@test quantile(d, 0.95) == 2

expected = [1/3 1 3/5 1/15]/2
@test_approx_eq logpdf(d, 0:3) log(expected)
@test_approx_eq pdf(d, 0:3) expected
@test_approx_eq cdf(d, 0:3) cumsum(expected,2)


# http://en.wikipedia.org/wiki/Fisher's_noncentral_hypergeometric_distribution
# (distributions are both equal to the (central) hypergeometric distribution when the odds ratio is 1.)
ns = 80
nf = 60
n = 100

d = Hypergeometric(ns,nf,n)

@test_approx_eq mean(d) 400.0/7
@test_approx_eq var(d) 48000.0/6811.0
@test_approx_eq skewness(d) sqrt(139/30)/92
@test_approx_eq kurtosis(d) 44952739/15124800-3
@test mode(d) == 57

@test quantile(d, 0.05) == 53
@test quantile(d, 0.95) == 62

expected = [0.10934614 0.13729286 0.14961947 0.14173721 0.11682889 0.08382473]
@test_approx_eq_eps logpdf(d, 55:60) log(expected) 1e-7
@test_approx_eq_eps pdf(d, 55:60) expected 1e-7
@test_approx_eq_eps cdf(d, 55:60) cumsum(expected,2)+cdf(d, 54) 1e-7
