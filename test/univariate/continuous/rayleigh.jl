@testset "edge cases" begin
    # issue #1371: cdf should not return -0.0
    @test cdf(Rayleigh(1), 0) === 0.0
    @test cdf(Rayleigh(1), -10) === 0.0
end
