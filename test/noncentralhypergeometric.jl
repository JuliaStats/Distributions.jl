using Distributions
using Base.Test

## Fisher's noncentral hypergeometric distribution
ns = 80
nf = 60
n = 100

# http://en.wikipedia.org/wiki/Fisher's_noncentral_hypergeometric_distribution
ω = 10.0
d = FisherNoncentralHypergeometric(ns, nf, n, ω)
@test d == typeof(d)(params(d)...)

@test isapprox(mean(d), 71.95759, atol=1e-5)
@test mode(d) == 72
@test quantile(d, 0.05) == 68
@test quantile(d, 0.95) == 75
@test isapprox(logpdf(d, 75), -2.600957  , atol=1e-6)
@test isapprox(pdf(d, 75)   ,  0.07420252, atol=1e-7)
@test isapprox(cdf(d, 75)   ,  0.9516117 , atol=1e-7)

ω = 0.361
d = FisherNoncentralHypergeometric(ns, nf, n, ω)

@test isapprox(mean(d), 50.42299, atol=1e-5)
@test mode(d) == 50
@test quantile(d, 0.05) == 47
@test quantile(d, 0.95) == 54
@test isapprox(logpdf(d, 50), -1.820071 , atol=1e-6)
@test isapprox(pdf(d, 50)   ,  0.1620142, atol=1e-7)
@test isapprox(cdf(d, 50)   ,  0.5203835, atol=1e-7)

# distributions are both equal to the (central) hypergeometric distribution when the odds ratio is 1.
ω = 1.0
d = FisherNoncentralHypergeometric(ns, nf, n, ω)

ref = Hypergeometric(ns,nf,n)
@test isapprox(mean(d)          , mean(ref)          , atol=1e-5)
@test isapprox(logpdf(d, 51)    , logpdf(ref, 51)    , atol=1e-7)
@test isapprox(pdf(d, 51)       , pdf(ref, 51)       , atol=1e-7)
@test isapprox(cdf(d, 51)       , cdf(ref, 51)       , atol=1e-7)
@test isapprox(quantile(d, 0.05), quantile(ref, 0.05), atol=1e-7)
@test isapprox(quantile(d, 0.95), quantile(ref, 0.95), atol=1e-7)
@test mode(d) == mode(ref)

## Wallenius' noncentral hypergeometric distribution
ns = 80
nf = 60
n = 100

# http://en.wikipedia.org/wiki/Fisher's_noncentral_hypergeometric_distribution
ω = 10.0
d = WalleniusNoncentralHypergeometric(ns, nf, n, ω)
@test d == typeof(d)(params(d)...)

@test isapprox(mean(d)      , 78.82945   , atol=1e-5)
@test mode(d)           == 80
@test quantile(d, 0.05) == 77
@test quantile(d, 0.95) == 80
@test isapprox(logpdf(d, 75), -4.750073  , atol=1e-6)
@test isapprox(pdf(d, 75)   ,  0.00865106, atol=1e-7)
@test isapprox(cdf(d, 75)   ,  0.01133378, atol=1e-7)

ω = 0.361
d = WalleniusNoncentralHypergeometric(ns, nf, n, ω)

@test isapprox(mean(d)      , 45.75323, atol=1e-5)
@test mode(d)           == 45
@test quantile(d, 0.05) == 42
@test quantile(d, 0.95) == 49
@test isapprox(logpdf(d, 50),-3.624478, atol=1e-6)
@test isapprox(pdf(d, 50)   , 0.026663, atol=1e-7)
@test isapprox(cdf(d, 50)   , 0.983674, atol=1e-7)

# distributions are both equal to the (central) hypergeometric distribution when the odds ratio is 1.
ω = 1.0
d = WalleniusNoncentralHypergeometric(ns, nf, n, ω)

ref = Hypergeometric(ns,nf,n)
@test isapprox(mean(d)          , mean(ref)          , atol=1e-5)
@test isapprox(logpdf(d, 51)    , logpdf(ref, 51)    , atol=1e-7)
@test isapprox(pdf(d, 51)       , pdf(ref, 51)       , atol=1e-7)
@test isapprox(cdf(d, 51)       , cdf(ref, 51)       , atol=1e-7)
@test isapprox(quantile(d, 0.05), quantile(ref, 0.05), atol=1e-7)
@test isapprox(quantile(d, 0.95), quantile(ref, 0.95), atol=1e-7)
@test mode(d) == mode(ref)

##### Conversions and Constructors
d = FisherNoncentralHypergeometric(ns, nf, n, 10.)
@test FisherNoncentralHypergeometric(ns, nf, n, 10) == d
@test typeof(convert(FisherNoncentralHypergeometric{Float32}, d)) == FisherNoncentralHypergeometric{Float32}
@test typeof(convert(FisherNoncentralHypergeometric{Float32}, ns, nf, n, 10.)) == FisherNoncentralHypergeometric{Float32}

d = WalleniusNoncentralHypergeometric(ns, nf, n, 10.)
@test WalleniusNoncentralHypergeometric(ns, nf, n, 10) == d
@test typeof(convert(WalleniusNoncentralHypergeometric{Float32}, d)) == WalleniusNoncentralHypergeometric{Float32}
@test typeof(convert(WalleniusNoncentralHypergeometric{Float32}, ns, nf, n, 10.)) == WalleniusNoncentralHypergeometric{Float32}
