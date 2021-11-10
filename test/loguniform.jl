module TestLogUniform
using Test
using Distributions
import Random

@testset "LogUniform" begin
    rng = Random.MersenneTwister(0)
    d = LogUniform(1,10)
    @test 1 <= rand(rng, d) <= 10
    @test typeof(rand(rng, d) ) == eltype(d)
    @test quantile(d, 0) ≈ 1
    @test quantile(d, 0.5) ≈ sqrt(10) # geomean
    @test quantile(d, 1) ≈ 10
    @test mode(d) ≈ 1
    @test insupport(d, 0) == false

    # numbers obtained by calling scipy.stats.loguniform
    @test std(d)  ≈ 2.49399867607628
    @test mean(d) ≈ 3.908650337129266
    @test pdf(d, 1.0001) ≈ 0.43425105679757203
    @test pdf(d, 5     ) ≈ 0.08685889638065035
    @test pdf(d, 9.9999) ≈ 0.04342988248915007
    @test cdf(d, 1.0001) ≈ 4.342727686266485e-05
    @test cdf(d, 5     ) ≈ 0.6989700043360187
    @test cdf(d, 9.9999) ≈ 0.999995657033466
    @test median(d) ≈ 3.1622776601683795
    @test logpdf(d, 5) ≈ -2.443470357682056

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


end#module
