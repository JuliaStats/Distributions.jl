using Distributions, Test

@testset "truncated uniform" begin
    # just test equivalence of truncation results
    u = Uniform(1, 2)
    @test truncated(u) === u
    @test truncated(u; lower=0) == u
    @test truncated(u; lower=0, upper=Inf) == u
    @test truncated(u; upper=2.1) == u
    @test truncated(u; lower=-Inf, upper=2.1) == u
    @test truncated(u; lower=1.1) == Uniform(1.1, 2)
    @test truncated(u; lower=1.1, upper=Inf) == Uniform(1.1, 2)
    @test truncated(u, 1.1, 2.1) == Uniform(1.1, 2)
    @test truncated(u, 1.1, 1.9) == Uniform(1.1, 1.9)
end
