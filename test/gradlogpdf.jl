using Distributions
using Test


# Test for gradlogpdf on univariate distributions

@test isapprox(gradlogpdf(Beta(1.5, 3.0), 0.7)    , -5.9523809523809526 , atol=1.0e-8)
@test isapprox(gradlogpdf(Chi(5.0), 5.5)          , -4.7727272727272725 , atol=1.0e-8)
@test isapprox(gradlogpdf(Chisq(7.0), 12.0)       , -0.29166666666666663, atol=1.0e-8)
@test isapprox(gradlogpdf(Exponential(2.0), 7.0)  , -0.5                , atol=1.0e-8)
@test isapprox(gradlogpdf(Gamma(9.0, 0.5), 11.0)  , -1.2727272727272727 , atol=1.0e-8)
@test isapprox(gradlogpdf(Gumbel(3.5, 1.0), 4.0)  , -0.3934693402873666 , atol=1.0e-8)
@test isapprox(gradlogpdf(Laplace(7.0), 34.0)     , -1.0                , atol=1.0e-8)
@test isapprox(gradlogpdf(Logistic(-6.0), 1.0)    , -0.9981778976111987 , atol=1.0e-8)
@test isapprox(gradlogpdf(LogNormal(5.5), 2.0)    ,  1.9034264097200273 , atol=1.0e-8)
@test isapprox(gradlogpdf(Normal(-4.5, 2.0), 1.6) , -1.525              , atol=1.0e-8)
@test isapprox(gradlogpdf(TDist(8.0), 9.1)        , -0.9018830525272548 , atol=1.0e-8)
@test isapprox(gradlogpdf(Weibull(2.0), 3.5)      , -6.714285714285714  , atol=1.0e-8)
@test isapprox(gradlogpdf(Uniform(-1.0, 1.0), 0.3),  0.0                , atol=1.0e-8)


# Test for gradlogpdf on multivariate distributions

@test isapprox(gradlogpdf(MvNormal([1., 2.], [1. 0.1; 0.1 1.]), [0.7, 0.9])   ,
    [0.191919191919192, 1.080808080808081]   ,atol=1.0e-8)
@test isapprox(gradlogpdf(MvTDist(5., [1., 2.], [1. 0.1; 0.1 1.]), [0.7, 0.9]),
    [0.2150711513583442, 1.2111901681759383] ,atol=1.0e-8)
