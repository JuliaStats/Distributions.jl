using Distributions, Test

@testset "truncated DiscreteUniform" begin
    # just test equivalence of truncation results
    bounds = [(1, 10), (-3, 7), (-5, -2)]
    @testset "lower=$lower, upper=$upper" for (lower, upper) in bounds
        d = DiscreteUniform(lower, upper)
        @test truncated(d, -Inf, Inf) == d
        @test truncated(d, nothing, nothing) === d
        @test truncated(d, lower - 0.1, Inf) == d
        @test truncated(d, lower - 0.1, nothing) == d
        @test truncated(d, -Inf, upper + 0.1) == d
        @test truncated(d, nothing, upper + 0.1) == d
        @test truncated(d, lower + 0.3, Inf) == DiscreteUniform(lower + 1, upper)
        @test truncated(d, lower + 0.3, nothing) == DiscreteUniform(lower + 1, upper)
        @test truncated(d, -Inf, upper - 0.5) == DiscreteUniform(lower, upper - 1)
        @test truncated(d, nothing, upper - 0.5) == DiscreteUniform(lower, upper - 1)
        @test truncated(d, lower + 1.5, upper - 1) == DiscreteUniform(lower + 2, upper - 1)
    end
end
