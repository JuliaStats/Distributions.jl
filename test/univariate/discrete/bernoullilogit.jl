using Distributions
using StatsFuns
using Test, Random

@testset "basic properties" begin
    @test BernoulliLogit() === BernoulliLogit(0.0)

    for logitp in (-0.3, 0.2, 0.1f0)
        d = BernoulliLogit(logitp)
        @test d isa BernoulliLogit{typeof(logitp)}
        @test convert(typeof(d), d) === d
        @test convert(BernoulliLogit{Float16}, d) === BernoulliLogit(Float16(logitp))
        @test eltype(typeof(d)) === Bool
        @test params(d) == (logitp,)
        @test partype(d) === typeof(logitp)
    end
end

@testset "succprob/failprob" begin
    for p in (0.0, 0.1, 0.31f0, 0.5, 0.7f0, 0.95, 1.0)
        d = BernoulliLogit(logit(p))
        @test @inferred(succprob(d)) ≈ p
        @test @inferred(failprob(d)) ≈ 1 - p
        @test @inferred(Distributions.logsuccprob(d)) ≈ log(p)
        @test @inferred(Distributions.logfailprob(d)) ≈ log1p(-p)
    end
end

@testset "rand" begin
    @test rand(BernoulliLogit()) isa Bool
    @test rand(BernoulliLogit(), 10) isa Vector{Bool}

    N = 10_000
    for p in (0.0, 0.1, 0.31f0, 0.5, 0.7f0, 0.95, 1.0)
        d = BernoulliLogit(logit(p))
        @test @inferred(rand(d)) isa Bool
        @test @inferred(rand(d, 10)) isa Vector{Bool}
        @test mean(rand(d, N)) ≈ p atol=0.01
    end
end

@testset "cgf" begin
    test_cgf(BernoulliLogit(), (1f0, -1f0, 1e6, -1e6))
    test_cgf(BernoulliLogit(0.1), (1f0, -1f0, 1e6, -1e6))
end

@testset "comparison with `Bernoulli`" begin
    for p in (0.0, 0.1, 0.31f0, 0.5, 0.7f0, 0.95, 1.0)
        d = BernoulliLogit(logit(p))
        d0 = Bernoulli(p)

        @test @inferred(mean(d)) ≈ mean(d0)
        @test @inferred(var(d)) ≈ var(d0)
        @test @inferred(skewness(d)) ≈ skewness(d0)
        @test @inferred(kurtosis(d)) ≈ kurtosis(d0)
        @test @inferred(mode(d)) ≈ mode(d0)
        @test @inferred(modes(d)) ≈ modes(d0)
        @test @inferred(median(d)) ≈ median(d0)
        @test @inferred(entropy(d)) ≈ entropy(d0)

        for x in (true, false, 0, 1, -3, 5)
            @test @inferred(pdf(d, x)) ≈ pdf(d0, x)
            @test @inferred(logpdf(d, x)) ≈ logpdf(d0, x)
            @test @inferred(cdf(d, x)) ≈ cdf(d0, x)
            @test @inferred(logcdf(d, x)) ≈ logcdf(d0, x)
            @test @inferred(ccdf(d, x)) ≈ ccdf(d0, x)
            @test @inferred(logccdf(d, x)) ≈ logccdf(d0, x)
        end

        for q in (-0.2f0, 0.25, 0.6f0, 1.5)
            @test @inferred(quantile(d, q)) ≈ quantile(d0, q) nans=true
            @test @inferred(cquantile(d, q)) ≈ cquantile(d0, q) nans=true
        end

        for t in (-5.2, 1.2f0)
            @test @inferred(mgf(d, t)) ≈ mgf(d0, t) rtol=1e-6
            @test @inferred(cgf(d, t)) ≈ cgf(d0, t) rtol=1e-6
            @test @inferred(cf(d, t)) ≈ cf(d0, t) rtol=1e-6
        end
    end
end
