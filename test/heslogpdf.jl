using Distributions
using Test

# Test for heslogpdf on univariate distributions

@test isapprox(heslogpdf(Beta(1.5, 3.0), 0.7)   , -23.242630385487523 , atol=1.0e-8)
@test isapprox(heslogpdf(Chi(5.0), 5.5)         , -1.1322314049586777 , atol=1.0e-8)
@test isapprox(heslogpdf(Chisq(7.0), 12.0)      , -0.01736111111111111, atol=1.0e-8)
@test isapprox(heslogpdf(Exponential(2.0), 7.0) ,  0.0                , atol=1.0e-8)
@test isapprox(heslogpdf(Frechet(2.0,4.0), 9.0) ,  0.02240512117055326, atol=1.0e-8)
@test isapprox(heslogpdf(Gamma(9.0, 0.5), 11.0) , -0.06611570247933884, atol=1.0e-8)
@test isapprox(heslogpdf(Gumbel(3.5, 1.0), 4.0) ,  0.6065306597126334 , atol=1.0e-8)
@test isapprox(heslogpdf(Laplace(7.0), 34.0)    ,  0.0                , atol=1.0e-8)
@test isapprox(heslogpdf(Logistic(-6.0), 1.0)   , -0.00182044236024365, atol=1.0e-8)
@test isapprox(heslogpdf(Logitnormal(3.0), 5.0) , -4.257529394046549  , atol=1.0e-8)
@test isapprox(heslogpdf(LogNormal(5.5), 2.0)   , -1.2017132048600137 , atol=1.0e-8)
@test isapprox(heslogpdf(Normal(-4.5, 2.0), 1.6), -0.25               , atol=1.0e-8)
@test isapprox(heslogpdf(TDist(8.0), 9.1)       ,  0.0816459812355031 , atol=1.0e-8)
@test isapprox(heslogpdf(Weibull(2.0), 3.5)     , -2.0816326530612246 , atol=1.0e-8)
@test isapprox(heslogpdf(Weibull(2.0), -3.5)    ,  0.0                , atol=1.0e-8)