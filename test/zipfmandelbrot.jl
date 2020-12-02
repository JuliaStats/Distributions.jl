using Test
using Distributions

@testset "ZipfMandelbrot" begin
    N, q, s= 100, 2, 3
    d = ZipfMandelbrot(N, q, s)
    @test params(d) == (100, 2.0, 3.0)
    @test params(ZipfMandelbrot(100.0, q, s)) == (100, 2.0, 3.0)
    @test partype(d) == Float64
    @test pdf(d, 2) ≈ 0.2028975363526503
    @test cdf(d, 2) ≈ 0.6838398447441176
    @test mean(d) ≈ 3.001707310467501
    @test mode(d) == 1
    @test entropy(d) ≈ 1.7696228967207608
end
