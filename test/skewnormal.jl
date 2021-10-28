using Test
using Distributions
import Distributions: normpdf, normcdf, normlogpdf, normlogcdf, cdf

@testset "SkewNormal" begin
    @test_throws ArgumentError SkewNormal(0.0, 0.0, 0.0)
    d1 = SkewNormal(1, 2, 3)
    d2 = SkewNormal(1.0f0, 2, 3)
    @test partype(d1) == Float64
    @test partype(d2) == Float32
    @test params(d1) == (1.0, 2.0, 3.0)
    # Azzalini sn: sprintf("%.17f",dsn(3.3, xi=1, omega=2, alpha=3))
    @test pdf(d1, 3.3) ≈ 0.20587854616839998
    @test minimum(d1) ≈ -Inf
    @test maximum(d1) ≈  Inf
    @test logpdf(d1, 3.3) ≈ log(pdf(d1, 3.3))
    ## cdf and quantile: when we get Owen's T
    #@test cdf(d, 4.5) ≈ 1.0 #when we get Owen's T
    @test mean(d1) ≈ 2.513879513212096
    @test std(d1) ≈ 1.306969326142243
    #
    d0 = SkewNormal(0.0, 1.0, 0.0)
    @test SkewNormal() == d0
    d3 = SkewNormal(0.5, 2.2, 0.0)
    d4 = Normal(0.5, 2.2)
    #
    @test pdf(d3, 3.3) == Distributions.pdf(d4, 3.3)
    @test pdf.(d3, 1:3) == Distributions.pdf.(d4, 1:3)
    a = mean(d3), var(d3), std(d3)
    b = Distributions.mean(d4), Distributions.var(d4), Distributions.std(d4)
    @test a == b
    @test skewness(d3) == Distributions.skewness(d4)
    @test kurtosis(d3) == Distributions.kurtosis(d4)
    @test mgf(d3, 2.25) == Distributions.mgf(d4, 2.25)
    @test cf(d3, 2.25) == Distributions.cf(d4, 2.25)
    # reference values computed with Mathematica
    gridx = [-0.15, 0.0, 0, 0.15] 
    cdfx = [
        0.003579417457235501,
        0.006369452573950052,
        0.006369452573950052,
        0.0108418482378097
    ]
    @test cdf.(d1, gridx) ≈ cdfx
end


# R code using Azzalini's package sn
#library("sn")
#sprintf("%.17f", dsn(3.3, xi=1, omega=2, alpha=3))
#f1 <- makeSECdistr(dp=c(1,2,3), family="SN", name="First-SN")
#sprintf("%.15f", mean(f1))
#sprintf("%.15f", sd(f1))
