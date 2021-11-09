module TestLogUniform
using Test
using Distributions
import Random

@testset "LogUniform" begin
    rng = Random.MersenneTwister(0)
    @testset "smoketests" begin
        d = LogUniform(1,10)
        @test 1 <= rand(rng, d) <= 10
        @test typeof(rand(rng, d) ) == eltype(d)
        @test quantile(d, 1) ≈ 10
        @test quantile(d, 0) ≈ 1
        @test mode(d) ≈ 1
        @test insupport(d, 0) == false

        @inferred mean(d)
        @inferred std(d)
        @inferred var(d)
        @inferred mode(d)
        @inferred modes(d)
        @inferred median(d)
        @inferred pdf(d, 1)
        @inferred cdf(d, 1)
        @inferred ccdf(d, 1)
        @inferred logpdf(d, 1)
        @inferred logcdf(d, 1)
        @inferred quantile(d, 1)
        @inferred cquantile(d, 1)
        @inferred rand(rng, d)
        @inferred params(d)
        @inferred partype(d)

        @test truncated(d, 2, 14) == LogUniform(2,10)
    end
end


end#module
