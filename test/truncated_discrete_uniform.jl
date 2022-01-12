using Distributions, Test

@testset "truncated DiscreteUniform" begin
    # just test equivalence of truncation results
    u = DiscreteUniform(1, 10)
    @test truncated(u, -Inf, Inf) == u
    @test truncated(u, 0, Inf) == u
    @test truncated(u, -Inf, 10.1) == u
    @test truncated(u, 1.1, Inf) == DiscreteUniform(2, 10)
    @test truncated(u, 1.1, 4.1) == DiscreteUniform(2, 4)
    @test truncated(u, 1.1, 3.9) == DiscreteUniform(2, 3)
end
