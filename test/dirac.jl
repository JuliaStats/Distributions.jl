using Distributions
using Test

@testset "Dirac tests" begin
    for val in [3, 3.0]
        d = Dirac(val)
        @test minimum(d) == val
        @test maximum(d) == val
        @test pdf(d, val - 1) == 0
        @test pdf(d, val) == 1
        @test pdf(d, val + 1) == 0
        @test cdf(d, val - 1) == 0
        @test cdf(d, val) == 1
        @test cdf(d, val + 1) == 1
        @test quantile(d, 0) == val
        @test quantile(d, 0.5) == val
        @test quantile(d, 1) == val
        @test rand(d) == val
    end
end
