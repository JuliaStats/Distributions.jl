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

# Test for gradlogpdf on univariate mixture distributions against centered finite-difference on logpdf

x = [-0.2, 0.3, 0.8, 1.3, 10.5]
delta = 0.001

for d in (
    MixtureModel([Normal(-4.5, 2.0)], [1.0]),
    MixtureModel([Exponential(2.0)], [1.0]),
    MixtureModel([Uniform(-1.0, 1.0)], [1.0]),
    MixtureModel([Normal(-2.0, 3.5), Normal(-4.5, 2.0)], [0.0, 1.0]),
    MixtureModel([Beta(1.5, 3.0), Chi(5.0), Chisq(7.0)], [0.4, 0.3, 0.3]),
    MixtureModel([Exponential(2.0), Gamma(9.0, 0.5), Gumbel(3.5, 1.0), Laplace(7.0)], [0.3, 0.2, 0.4, 0.1]),
    MixtureModel([Logistic(-6.0), LogNormal(5.5), TDist(8.0), Weibull(2.0)], [0.3, 0.2, 0.4, 0.1])
)
    xs = filter(s -> insupport(d, s), x)
    glp1 = gradlogpdf(d, xs)
    glp2 = ( logpdf.(d, xs .+ delta) - logpdf.(d, xs .- delta) ) ./ 2delta
    @test isapprox(glp1, glp2, atol = delta)
end

# Test for gradlogpdf on multivariate mixture distributions against centered finite-difference on logpdf

x = [[0.2, 0.3], [0.8, 1.3], [-1.0, 10.5]]
delta = 0.001

for d in (
    MixtureModel([MvNormal([1., 2.], [1. 0.1; 0.1 1.])], [1.0]),
    MixtureModel([MvNormal([1.0, 2.0], [0.4 0.2; 0.2 0.5]), MvNormal([2.0, 1.0], [0.3 0.1; 0.1 0.4])], [0.4, 0.6]),
    MixtureModel([MvNormal([1.0, 2.0], [0.4 0.2; 0.2 0.5]), MvNormal([2.0, 1.0], [0.3 0.1; 0.1 0.4])], [1.0, 0.0]),
    MixtureModel([MvTDist(5., [1., 2.], [1. 0.1; 0.1 1.])], [1.0]),
    MixtureModel([MvNormal([1.0, 2.0], [0.4 0.2; 0.2 0.5]), MvTDist(5., [1., 2.], [1. 0.1; 0.1 1.])], [0.4, 0.6])
)
    xs = filter(s -> insupport(d, s), x)
    glp = gradlogpdf(d, xs)
    glpx = ( logpdf(d, xs .+ [[delta, 0]]) - logpdf(d, xs .- [[delta, 0]]) ) ./ 2delta
    glpy = ( logpdf(d, xs .+ [[0, delta]]) - logpdf(d, xs .- [[0, delta]]) ) ./ 2delta
    @test isapprox(getindex.(glp, 1), glpx, atol=delta)
    @test isapprox(getindex.(glp, 2), glpy, atol=delta)
end
