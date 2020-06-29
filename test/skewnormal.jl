using Test
using Distributions
import Distributions: normpdf, normcdf, normlogpdf, normlogcdf

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
end

# R code using Azzalini's package sn
#library("sn")
#sprintf("%.17f", dsn(3.3, xi=1, omega=2, alpha=3))
#f1 <- makeSECdistr(dp=c(1,2,3), family="SN", name="First-SN")
#sprintf("%.15f", mean(f1))
#sprintf("%.15f", sd(f1))
