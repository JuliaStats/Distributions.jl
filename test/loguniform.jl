module TestLogUniform
using Test
using Distributions
import Random

@testset "LogUniform" begin
    rng = Random.MersenneTwister(0)

    @test pdf(LogUniform(1f0, 2f0), 1) isa Float32
    @test pdf(LogUniform(1, 2), 1f0) isa Float32
    @test pdf(LogUniform(1, 2), 1) isa Float64
    @test quantile(LogUniform(1, 2), 1) isa Float64
    @test quantile(LogUniform(1, 2), 1f0) isa Float32
    @testset "$f" for f in [pdf, cdf, quantile, logpdf, logcdf]
        @test @inferred(f(LogUniform(1,2), 1)) isa Float64
        @test @inferred(f(LogUniform(1,2), 1.0)) isa Float64
        @test @inferred(f(LogUniform(1.0,2), 1.0)) isa Float64
        @test @inferred(f(LogUniform(1.0f0,2), 1)) isa Float32
        @test @inferred(f(LogUniform(1.0f0,2), 1f0)) isa Float32
        @test @inferred(f(LogUniform(1,2), 1f0)) isa Float32
    end

    d = LogUniform(1,10)
    @test eltype(d) === Float64
    @test 1 <= rand(rng, d) <= 10
    @test rand(rng, d) isa eltype(d)
    @test @inferred(quantile(d, 0))   ≈ 1
    @test quantile(d, 0.5) ≈ sqrt(10) # geomean
    @test quantile(d, 1)   ≈ 10
    @test mode(d) ≈ 1
    @test !insupport(d, 0)
    @test @inferred(minimum(d)) === 1
    @test @inferred(maximum(d)) === 10
    @test partype(d) === Int
    @test truncated(d, 2, 14) === LogUniform(2,10)

    # numbers obtained by calling scipy.stats.loguniform
    @test @inferred(std(d)        ) ≈ 2.49399867607628
    @test @inferred(mean(d)       ) ≈ 3.908650337129266
    @test @inferred(pdf(d, 1.0001)) ≈ 0.43425105679757203
    @test @inferred(pdf(d, 5     )) ≈ 0.08685889638065035
    @test @inferred(pdf(d, 9.9999)) ≈ 0.04342988248915007
    @test @inferred(cdf(d, 1.0001)) ≈ 4.342727686266485e-05
    @test @inferred(cdf(d, 5     )) ≈ 0.6989700043360187
    @test @inferred(cdf(d, 9.9999)) ≈ 0.999995657033466
    @test @inferred(median(d)     ) ≈ 3.1622776601683795
    @test @inferred(logpdf(d, 5)  ) ≈ -2.443470357682056

    for _ in 1:10
        lo = rand(rng)
        hi = lo + 10*rand(rng)
        dist = LogUniform(lo,hi)
        q = rand(rng)
        @test cdf(dist, quantile(dist, q)) ≈ q

        u = Uniform(log(lo), log(hi))
        @test exp(quantile(u, q)) ≈ quantile(dist, q)
        @test exp(median(u)) ≈ median(dist)
        x = rand(rng, dist)
        @test cdf(u, log(x)) ≈ cdf(dist, x)
    end
end


end#module
