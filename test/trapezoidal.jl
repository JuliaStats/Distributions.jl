using Test
using Distributions

@testset "Trapezoidal" begin
    @test_throws ArgumentError TrapezoidalDist(2,1,3,4)
    @test_throws ArgumentError TrapezoidalDist(1,3,2,4)
    @test_throws ArgumentError TrapezoidalDist(1,2,4,3)

    d = TrapezoidalDist(1, 2, 3, 4)
    d2 = TrapezoidalDist(1.0f0, 2, 3, 4)
    @test partype(d) == Float64
    @test partype(d2) == Float32
    @test params(d) == (1.0, 2.0, 3.0, 4.0)
    @test minimum(d) ≈ 1
    @test maximum(d) ≈ 4

    @test logpdf(d, 3.3) ≈ log(pdf(d, 3.3))
    @test logpdf(d, 2) == logpdf(d, 3)
    # out of support
    @test isinf(logpdf(d, 0.5))
    @test isinf(logpdf(d, 4.5))
    @test cdf(d, 0.5) ≈ 0.0
    @test cdf(d, 4.5) ≈ 1.0
    # on support limits
    @test isinf(logpdf(d, 1))
    @test isinf(logpdf(d, 4))
    @test cdf(d, 1) ≈ 0.0
    @test cdf(d, 4) ≈ 1.0
    @test cdf(d, 2.5) ≈ 1/2

    @test mean(d) ≈ 2.5
    @test var(d) ≈ 0.41666666666666696
end
