using Distributions
using Base.Test

## Fisher's noncentral hypergeometric distribution
ns = 80
nf = 60
n = 100

# http://en.wikipedia.org/wiki/Fisher's_noncentral_hypergeometric_distribution
ω = 10.0
d = FisherNoncentralHypergeometric(ns, nf, n, ω)

@test_approx_eq_eps mean(d) 71.95759 1e-5
@test mode(d) == 72
@test quantile(d, 0.05) == 68
@test quantile(d, 0.95) == 75
@test_approx_eq_eps logpdf(d, 75) -2.600957 1e-6
@test_approx_eq_eps pdf(d, 75) 0.07420252 1e-7
@test_approx_eq_eps cdf(d, 75) 0.9516117 1e-7

ω = 0.361
d = FisherNoncentralHypergeometric(ns, nf, n, ω)

@test_approx_eq_eps mean(d) 50.42299 1e-5
@test mode(d) == 50
@test quantile(d, 0.05) == 47
@test quantile(d, 0.95) == 54
@test_approx_eq_eps logpdf(d, 50) -1.820071 1e-6
@test_approx_eq_eps pdf(d, 50) 0.1620142 1e-7
@test_approx_eq_eps cdf(d, 50) 0.5203835 1e-7

# distributions are both equal to the (central) hypergeometric distribution when the odds ratio is 1.
ω = 1.0
d = FisherNoncentralHypergeometric(ns, nf, n, ω)

ref = Hypergeometric(ns,nf,n)
@test_approx_eq_eps mean(d) mean(ref) 1e-5
@test_approx_eq_eps logpdf(d, 51) logpdf(ref, 51) 1e-7
@test_approx_eq_eps pdf(d, 51) pdf(ref, 51) 1e-7
@test_approx_eq_eps cdf(d, 51) cdf(ref, 51) 1e-7
@test_approx_eq_eps quantile(d, 0.05) quantile(ref, 0.05) 1e-7
@test_approx_eq_eps quantile(d, 0.95) quantile(ref, 0.95) 1e-7
@test mode(d) == mode(ref)

## Wallenius' noncentral hypergeometric distribution
ns = 80
nf = 60
n = 100

# http://en.wikipedia.org/wiki/Fisher's_noncentral_hypergeometric_distribution
ω = 10.0
d = WalleniusNoncentralHypergeometric(ns, nf, n, ω)

@test_approx_eq_eps mean(d) 78.82945 1e-5
@test mode(d) == 80
@test quantile(d, 0.05) == 77
@test quantile(d, 0.95) == 80
@test_approx_eq_eps logpdf(d, 75) -4.750073 1e-6
@test_approx_eq_eps pdf(d, 75) 0.00865106 1e-7
@test_approx_eq_eps cdf(d, 75) 0.01133378 1e-7

ω = 0.361
d = WalleniusNoncentralHypergeometric(ns, nf, n, ω)

@test_approx_eq_eps mean(d) 45.75323 1e-5
@test mode(d) == 45
@test quantile(d, 0.05) == 42
@test quantile(d, 0.95) == 49
@test_approx_eq_eps logpdf(d, 50) -3.624478 1e-6
@test_approx_eq_eps pdf(d, 50) 0.026663 1e-7
@test_approx_eq_eps cdf(d, 50) 0.983674 1e-7

# distributions are both equal to the (central) hypergeometric distribution when the odds ratio is 1.
ω = 1.0
d = WalleniusNoncentralHypergeometric(ns, nf, n, ω)

ref = Hypergeometric(ns,nf,n)
@test_approx_eq_eps mean(d) mean(ref) 1e-5
@test_approx_eq_eps logpdf(d, 51) logpdf(ref, 51) 1e-7
@test_approx_eq_eps pdf(d, 51) pdf(ref, 51) 1e-7
@test_approx_eq_eps cdf(d, 51) cdf(ref, 51) 1e-7
@test_approx_eq_eps quantile(d, 0.05) quantile(ref, 0.05) 1e-7
@test_approx_eq_eps quantile(d, 0.95) quantile(ref, 0.95) 1e-7
@test mode(d) == mode(ref)