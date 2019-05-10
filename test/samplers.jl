# Testing of samplers

using Distributions
using Distances
using Test


import Distributions:
    CategoricalDirectSampler,
    AliasTable,
    BinomialGeomSampler,
    BinomialTPESampler,
    BinomialPolySampler,
    BinomialAliasSampler,
    PoissonADSampler,
    PoissonCountSampler,
    ExponentialSampler,
    GammaGDSampler,
    GammaGSSampler,
    GammaMTSampler,
    GammaIPSampler,
    PoissBinAliasSampler,
    WeightedResampler

n_tsamples = 10^6

## Categorical samplers

@test_throws ArgumentError CategoricalDirectSampler(Float64[])
@test_throws ArgumentError AliasTable(Float64[])

rng = MersenneTwister(123)
for S in [CategoricalDirectSampler, AliasTable]
    local S
    println("    testing $S")
    for p in Any[[1.0], [0.3, 0.7], [0.2, 0.3, 0.4, 0.1]]
        test_samples(S(p), Categorical(p), n_tsamples)
        test_samples(S(p), Categorical(p), n_tsamples, rng=rng)
    end
end


## Binomial samplers

binomparams = [(0, 0.4), (0, 0.6), (5, 0.0), (5, 1.0),
               (1, 0.2), (1, 0.8), (3, 0.4), (4, 0.6),
               (40, 0.5), (100, 0.4), (300, 0.6)]

for (S, paramlst) in [
    (BinomialGeomSampler, [(0, 0.4), (0, 0.6), (5, 0.0), (5, 1.0), (1, 0.2), (1, 0.8), (3, 0.4), (4, 0.6)]),
    (BinomialTPESampler, [(40, 0.5), (100, 0.4), (300, 0.6)]),
    (BinomialPolySampler, binomparams),
    (BinomialAliasSampler, binomparams) ]
    local S
    println("    testing $S")
    for pa in paramlst
        n, p = pa
        test_samples(S(n, p), Binomial(n, p), n_tsamples)
        test_samples(S(n, p), Binomial(n, p), n_tsamples, rng=rng)
    end
end


## Poisson samplers

for (S, paramlst) in [
    (PoissonCountSampler, [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0]),
    (PoissonADSampler, [5.0, 10.0, 15.0, 20.0, 30.0])]
    local S

    println("    testing $S")
    for μ in paramlst
        test_samples(S(μ), Poisson(μ), n_tsamples)
        test_samples(S(μ), Poisson(μ), n_tsamples, rng=rng)
    end
end

## Poisson Binomial sampler
S = PoissBinAliasSampler
paramlst = (fill(0.2, 30), range(0.1, stop = .99, length = 30), [fill(0.1, 10); fill(0.9, 10)])
println("    testing $S")
for p in paramlst
    d = PoissonBinomial(p)
    test_samples(S(d), d, n_tsamples)
    test_samples(S(d), d, n_tsamples, rng=rng)
end


## Exponential samplers

for S in [ExponentialSampler]
    local S
    println("    testing $S")
    for scale in [1.0, 2.0, 3.0]
        test_samples(S(scale), Exponential(scale), n_tsamples)
        test_samples(S(scale), Exponential(scale), n_tsamples, rng=rng)
    end
end


## Gamma samplers
# shape >= 1
for S in [GammaGDSampler, GammaMTSampler]
    local S
    println("    testing $S")
    for d in [Gamma(1.0, 1.0), Gamma(2.0, 1.0), Gamma(3.0, 1.0),
               Gamma(1.0, 2.0), Gamma(3.0, 2.0), Gamma(100.0, 2.0)]
               test_samples(S(d), d, n_tsamples)
               test_samples(S(d), d, n_tsamples, rng=rng)
    end
end

# shape < 1
for S in [GammaGSSampler, GammaIPSampler]
    local S
    println("    testing $S")
    for d in [Gamma(0.1,1.0),Gamma(0.9,1.0)]
        test_samples(S(d), d, n_tsamples)
        test_samples(S(d), d, n_tsamples, rng=rng)
    end
end

@testset "WeightedResampler" begin
    rng = MersenneTwister(1234)

    @testset "Univariate" begin
        obs = collect(1:12)

        @testset "Equally Weighted" begin
            # Constant analytic weights
            wv = aweights(ones(12))

            s = WeightedResampler(obs, wv)
            X = rand(rng, s, 100000)

            # The mean values of the samples should roughly match the mean of the
            # original observation
            @test isapprox(mean(X), mean(obs); atol=0.01)
        end

        @testset "Linearly Weighted" begin
            # Linearly increasing analytic weights
            wv = aweights(collect(1/12:1/12:1.0))

            s = WeightedResampler(obs, wv)
            X = rand(rng, s, 100000)

            # The mean of the samples should not match the mean of the
            # original observation
            @test !isapprox(mean(X), mean(obs); atol=0.01)

            # 12 should be sampled the most
            @test mode(X) == 12
        end
    end

    @testset "Multivariate" begin
        v = [1.2, 0.7, -0.3, 5.4, -2.8]
        # Define different observations via arbitrary operations on v
        obs = hcat(
            v, reverse(v), sort(v), sin.(v), cos.(v), tan.(v),
            v / 100, v * 2, abs.(v), log.(abs.(v)), v .^ 2, v * 10,
        )

        @testset "Equally Weighted" begin
            # Constant analytic weights
            wv = aweights(ones(12))

            s = WeightedResampler(obs, wv)
            X = rand(rng, s, 100000)

            # The mean values of each variable in the samples should roughly match
            # the means of the original observation
            @test nrmsd(mean(X; dims=2), mean(obs; dims=2)) < 0.001
        end

        @testset "Linearly Weighted" begin
            # Linearly increasing analytic weights
            wv = aweights(collect(0.083:0.083:1.0))

            s = WeightedResampler(obs, wv)
            X = rand(rng, s, 100000)

            # The mean values of each variable of the samples should not match the
            # means of the original observation
            @test nrmsd(mean(X; dims=2), mean(obs; dims=2)) > 0.1

            # v * 10 should be sampled the most
            @test vec(mapslices(mode, X; dims=2)) == v * 10
        end
    end
    @testset "Matrixvariate" begin
        # NOTE: Since we've already testing the sampling behaviour we just want to
        # check that we've implement the Distributions API correctly for the
        # Matrxivariate case
        s = WeightedResampler([rand(4, 3) for i in 1:10], aweights(rand(10)))
        X = rand(s)
    end

    @testset "DimensionMismatch" begin
        @test_throws DimensionMismatch WeightedResampler(rand(10), aweights(collect(1:12)))
    end
end
