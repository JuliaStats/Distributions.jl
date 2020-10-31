using Test
using Distributions
import Distributions: tdistpdf, tdistlogpdf, tdistcdf

@testset "SkewTDist" begin
    @test_throws ArgumentError SkewTDist(0.0, 0.0, 0.0, 0.0)
    d1 = SkewTDist(1, 2, 3, 4)
    d2 = SkewTDist(1.0f0, 2, 3, 4)
    @test partype(d1) == Float64
    @test partype(d2) == Float32
    @test params(d1) == (1.0, 2.0, 3.0, 4.0)
    #
    @test pdf(d1, 1.1) ≈ 0.21090283222751241
    @test minimum(d1) ≈ -Inf
    @test maximum(d1) ≈  Inf
    @test logpdf(d1, 1.1) ≈ log(pdf(d1, 1.1))
    # CDF currently uses QuadGK
    @test cdf(d1, 1.1) ≈ 0.12234583048662456
    ## quantile: depends on Roots.jl
    @test mean(d1) ≈ 2.897366596101028
    @test std(d1) ≈ 2.097617696340303
    #
    d0 = SkewTDist(0.0, 1.0, 0.0, 5.71)
    @test SkewTDist(5.71) == d0
    d3 = SkewTDist(0.0, 1.0, 0.0, 5.71)
    d4 = TDist(5.71)
    #
    @test pdf(d3, 3.3) == Distributions.pdf(d4, 3.3)
    @test pdf.(d3, 1:3) == Distributions.pdf.(d4, 1:3)
    # cdf uses QuadGK
    @test cdf(d3, 3.3) ≈ Distributions.cdf(d4, 3.3)
    @test cdf.(d3, 1:3) ≈ Distributions.cdf.(d4, 1:3)
    #
    a = mean(d3), var(d3), std(d3)
    b = Distributions.mean(d4), Distributions.var(d4), Distributions.std(d4)
    @test a == b
    #
    @test skewness(d3) == Distributions.skewness(d4)
    # Kurtosis needs to be implemented.
    #@test kurtosis(d3) == Distributions.kurtosis(d4)
end




# R code using Azzalini's package sn
# library("sn")
# sprintf("%.17f", dst(1.1, xi=1, omega=2, alpha=3, nu=4))
# sprintf("%.17f", pst(1.1, xi=1, omega=2, alpha=3, nu=4))
# f1 <- makeSECdistr(dp=c(1,2,3,4), family="ST")
# sprintf("%.15f", mean(f1))
# sprintf("%.15f", sd(f1))
