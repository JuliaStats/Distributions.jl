# Testing of samplers

using Distributions, Random, StatsBase, Test

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
    PoissBinAliasSampler

@testset "samplers" begin
    n_tsamples = 10^6

    ## Categorical samplers

    @test_throws ArgumentError CategoricalDirectSampler(Float64[])
    @test_throws ArgumentError AliasTable(Float64[])

    rng = MersenneTwister(123)
    @testset "Categorical: $S" for S in [CategoricalDirectSampler, AliasTable]
        @testset "p=$p" for p in Any[[1.0], [0.3, 0.7], [0.2, 0.3, 0.4, 0.1]]
            test_samples(S(p), Categorical(p), n_tsamples)
            test_samples(S(p), Categorical(p), n_tsamples, rng=rng)
            @test ncategories(S(p)) == length(p)
        end
    end

    @test string(AliasTable(Float16[1,2,3])) == "AliasTable with 3 entries"

    ## Binomial samplers

    binomparams = [(0, 0.4), (0, 0.6), (5, 0.0), (5, 1.0),
                   (1, 0.2), (1, 0.8), (3, 0.4), (4, 0.6),
                   (40, 0.5), (100, 0.4), (300, 0.6)]

    @testset "Binomial: $S" for (S, paramlst) in [
            (BinomialGeomSampler, [(0, 0.4), (0, 0.6), (5, 0.0), (5, 1.0), (1, 0.2), (1, 0.8), (3, 0.4), (4, 0.6)]),
            (BinomialTPESampler, [(40, 0.5), (100, 0.4), (300, 0.6)]),
            (BinomialPolySampler, binomparams),
            (BinomialAliasSampler, binomparams) ]
        @testset "pa=$pa" for pa in paramlst
            n, p = pa
            test_samples(S(n, p), Binomial(n, p), n_tsamples)
            test_samples(S(n, p), Binomial(n, p), n_tsamples, rng=rng)
        end
    end

    ## Poisson samplers

    @testset "Poisson: $S" for
        (S, paramlst) in [
            (PoissonCountSampler, [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0]),
            (PoissonADSampler, [10.0, 15.0, 20.0, 30.0])]
        @testset "μ=$μ" for μ in paramlst
            test_samples(S(μ), Poisson(μ), n_tsamples)
            test_samples(S(μ), Poisson(μ), n_tsamples, rng=rng)
        end
    end

    ## Poisson Binomial sampler
    @testset "Poisson-Binomial: $S" for S in (PoissBinAliasSampler,)
        @testset "p=$p" for p in (fill(0.2, 30), range(0.1, stop = .99, length = 30), [fill(0.1, 10); fill(0.9, 10)])
            d = PoissonBinomial(p)
            test_samples(S(d), d, n_tsamples)
            test_samples(S(d), d, n_tsamples, rng=rng)
        end
    end


    ## Exponential samplers

    @testset "Exponential: $S" for S in [ExponentialSampler]
        @testset "scale=$scale" for scale in [1.0, 2.0, 3.0]
            test_samples(S(scale), Exponential(scale), n_tsamples)
            test_samples(S(scale), Exponential(scale), n_tsamples, rng=rng)
        end
    end


    ## Gamma samplers
    @testset "Gamma (shape >= 1): $S" for S in [GammaGDSampler, GammaMTSampler]
        @testset "d=$d" for d in [Gamma(1.0, 1.0), Gamma(2.0, 1.0), Gamma(3.0, 1.0),
              Gamma(1.0, 2.0), Gamma(3.0, 2.0), Gamma(100.0, 2.0)]
            test_samples(S(d), d, n_tsamples)
            test_samples(S(d), d, n_tsamples, rng=rng)
        end
    end

    @testset "Gamma (shape < 1): $S" for S in [GammaGSSampler, GammaIPSampler]
        @testset "d=$d" for d in [Gamma(0.1,1.0),Gamma(0.9,1.0)]
            test_samples(S(d), d, n_tsamples)
            test_samples(S(d), d, n_tsamples, rng=rng)
        end
    end

    @testset "GammaIPSampler" begin
        @testset "d=$d" for d in [Gamma(0.1, 1.0), Gamma(0.9, 1.0)]
            s = sampler(d)
            @test s isa GammaIPSampler{<:GammaMTSampler}
            @test s.s isa GammaMTSampler
            test_samples(s, d, n_tsamples)
            test_samples(s, d, n_tsamples, rng=rng)

            s = @inferred(GammaIPSampler(d, GammaMTSampler))
            @test s isa GammaIPSampler{<:GammaMTSampler}
            @test s.s isa GammaMTSampler
            test_samples(s, d, n_tsamples)
            test_samples(s, d, n_tsamples, rng=rng)

            s = @inferred(GammaIPSampler(d, GammaGDSampler))
            @test s isa GammaIPSampler{<:GammaGDSampler}
            @test s.s isa GammaGDSampler
            test_samples(s, d, n_tsamples)
            test_samples(s, d, n_tsamples, rng=rng)
        end
    end

    @testset "Random.Sampler" begin
        for dist in (
            Binomial(5, 0.3),
            Exponential(2.0),
            Gamma(0.1, 1.0),
            Gamma(2.0, 1.0),
            MatrixNormal(3, 4),
            MvNormal(zeros(3), I),
            Normal(1.5, 2.0),
            Poisson(0.5),
        )
            @test Random.Sampler(rng, dist, Val(1)) == dist
            @test Random.Sampler(rng, dist) == sampler(dist)
        end
    end
end
