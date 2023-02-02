using Distributions
using Test

@testset "triangular" begin
    a1 = 0.0
    b1 = 1.0
    c1 = 1.0
    x1 = 0.0
    x2 = 0.5
    x3 = 1.0

    d1 = TriangularDist(a1, b1, c1)
    @test mean(d1) == 2 / 3
    @test var(d1) == 1 / 18
    @test pdf(d1, x1) == 2 * x1
    @test pdf(d1, x2) == 2 * x2
    @test pdf(d1, x3) == 2 * x3
    @test cdf(d1, x1) == x1 ^ 2
    @test cdf(d1, x2) == x2 ^ 2
    @test cdf(d1, x3) == x3 ^ 2
end
