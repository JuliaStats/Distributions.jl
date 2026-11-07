#

using Distributions
using Test


dg = Gamma(1,1)

d_s = EdgeworthSum(dg,10)
d_m = EdgeworthMean(dg,10)
d_z = EdgeworthZ(dg,10)

dg_s = Gamma(10,1)
dg_m = Gamma(10,0.1)
dg_za = Gamma(10,1/std(dg_s))

for i = 0.01:0.01:0.99

    q = quantile(dg_s,i)
    @test isapprox(quantile(d_s, i)     , q          , atol=0.02)
    @test isapprox(cquantile(d_s, 1 - i), q          , atol=0.02)
    @test isapprox(cdf(d_s, q)          , i          , atol=0.002)
    @test isapprox(ccdf(d_s, q)         , 1 - i      , atol=0.002)
    @test isapprox(pdf(d_s, q)          , pdf(dg_s,q), atol=0.005)

    q = quantile(dg_m,i)
    @test isapprox(quantile(d_m, i)     , q          , atol=0.01)
    @test isapprox(cquantile(d_m, 1 - i), q          , atol=0.01)
    @test isapprox(cdf(d_m, q)          , i          , atol=0.002)
    @test isapprox(ccdf(d_m, q)         , 1 - i      , atol=0.002)
    @test isapprox(pdf(d_m, q)          , pdf(dg_m,q), atol=0.05)

    q = quantile(dg_za,i) - mean(dg_za)
    @test isapprox(quantile(d_z, i)     , q                       , atol=0.01)
    @test isapprox(cquantile(d_z, 1 - i), q                       , atol=0.01)
    @test isapprox(cdf(d_z, q)          , i                       , atol=0.002)
    @test isapprox(ccdf(d_z, q)         , 1 - i                   , atol=0.002)
    @test isapprox(pdf(d_z, q)          , pdf(dg_za,q+mean(dg_za)), atol=0.02)

end

@testset "support" begin
    @testset "$(nameof(typeof(d)))" for d in (d_s, d_m, d_z)
        @test minimum(d) == -Inf
        @test maximum(d) == Inf
        @test extrema(d) == (-Inf, Inf)
        @test insupport(d, -1.0)
    end
    # `dg` is non-negative, and so are its sum and its mean, but the expansion is not
    @test minimum(dg) == 0
    @test pdf(d_s, -1.0) > 0
    @test cdf(d_s, -1.0) > 0
    @test pdf(d_m, -1.0) > 0
    @test cdf(d_m, -1.0) > 0
end



