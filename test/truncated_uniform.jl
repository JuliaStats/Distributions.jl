using Distributions, Test

@testset "truncated uniform" begin
    # just test equivalence of truncation results
    u = Uniform(1, 2)
    @test Truncated(u, -Inf, Inf) == u
    @test Truncated(u, 0, Inf) == u
    @test Truncated(u, -Inf, 2.1) == u
    @test Truncated(u, 1.1, Inf) == Uniform(1.1, 2)
    @test Truncated(u, 1.1, 2.1) == Uniform(1.1, 2)
    @test Truncated(u, 1.1, 1.9) == Uniform(1.1, 1.9)
end
