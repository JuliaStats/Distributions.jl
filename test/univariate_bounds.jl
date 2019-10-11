using Distributions
using InteractiveUtils
using Test

dists = subtypes(UnivariateDistribution)
filter!(x -> hasmethod(x, ()), dists)
filter!(x -> isbounded(x()), dists)

@testset "bound checking $dist" for dist in dists
    d = dist()
    lb,ub = float.(extrema(support(d)))
    lb = prevfloat(lb)
    ub = nextfloat(ub)
    @test iszero(cdf(d, lb))
    @test isone(cdf(d, ub))

    lb_lcdf = logcdf(d,lb)
    @test isinf(lb_lcdf) & (lb_lcdf < 0)
    @test iszero(logcdf(d, ub))

    @test isone(ccdf(d, lb))
    @test iszero(ccdf(d, ub))

    ub_lccdf = logccdf(d,ub)
    @test isinf(ub_lccdf) & (ub_lccdf < 0)
    @test iszero(logccdf(d, lb))

    @test iszero(pdf(d, lb))
    @test iszero(pdf(d, ub))

    lb_lpdf = logpdf(d, lb)
    @test isinf(lb_lpdf) & (lb_lpdf < 0)
    ub_lpdf = logpdf(d, ub)
    @test isinf(ub_lpdf) & (ub_lpdf < 0)
end
