using Distributions
using Test, Random, ForwardDiff
using ForwardDiff: Dual
using Random
using Random: GLOBAL_RNG
using Unitful

function test_types(D::Type{<:UnivariateDistribution{<:ContinuousSupport}},
                    targs)
    dist = D(targs...)
    if length(targs) > 0
        @test eltype(dist) ≡ eltype(targs)
    end
    @test eltype(dist) ≡ typeof(rand(dist)) ≡
        typeof(rand(GLOBAL_RNG, dist)) ≡ eltype(rand(dist, 3))
    @test eltype(dist) ≡ typeof(mean(dist)) ≡ typeof(median(dist)) ≡
        typeof(mode(dist)) ≡ typeof(std(dist))
    @test eltype(dist) ≡
        typeof(quantile(dist, 0.5)) ≡ typeof(cquantile(dist, 0.5)) ≡
        typeof(invlogcdf(dist, -2.0)) ≡ typeof(invlogccdf(dist, -2.0))
    @test typeof(var(dist)) ≡ typeof(mean(dist) * mean(dist))
    @test typeof(skewness(dist)) ≡ typeof(kurtosis(dist)) ≡
        typeof(one(eltype(dist)))
end

function test_types(D::Type{<:UnivariateDistribution{<:CountableSupport}},
                    targs)
    dist = D(targs...)
    @test eltype(dist) ≡ typeof(rand(dist)) ≡
        typeof(rand(GLOBAL_RNG, dist)) ≡ eltype(rand(dist, 3))
    @test eltype(dist) ≡ typeof(mode(dist))
    @test eltype(dist) ≡
        typeof(quantile(dist, 0.5)) ≡ typeof(cquantile(dist, 0.5)) ≡
        typeof(invlogcdf(dist, -2.0)) ≡ typeof(invlogccdf(dist, -2.0))
    @test typeof(var(dist)) ≡ typeof(mean(dist) * mean(dist))
    @test eltype(support(dist)) ≡ typeof(one(eltype(dist)))
end

@testset "Support for continuous distributions supporting just Float64" begin
    @testset "Testing $Dist" for (Dist, args) in [
        Cosine => (),
        Cosine => (1.5,),
        Cosine => (2.0, 2.8),
        # These should be broader!
        Arcsine => (),
        Arcsine => (2.0,),
        Arcsine => (1.0, 2.5),
        Arcsine => (-1.0, 1.0),
        Beta => (2.0, 2.0),
        Beta => (3.0, 4.0),
        Beta => (17.0, 13.0),
        # BetaPrime => (),
        # BetaPrime => (3.0,),
        # BetaPrime => (3.0, 5.0),
        # BetaPrime => (5.0, 3.0),
        Biweight => (0.0, 1.0),
        Cauchy => (),
        Cauchy => (2.0,),
        Cauchy => (0.0, 1.0),
        Cauchy => (10.0, 1.0),
        Cauchy => (2.0, 10.0),
        # Chernoff => (),
        Chi => (1,),
        Chi => (2,),
        Chi => (3,),
        Chi => (12,),
        # Chisq => (1,),
        # Chisq => (8,),
        # Chisq => (20,),
        Epanechnikov => (0.0, 1.0),
        Erlang => (),
        # Erlang has to be handled separately => (3,),
        # Erlang has to be handled separately => (3, 1.0),
        # Erlang has to be handled separately => (5, 2.0),
        Exponential => (),
        Exponential => (2.0,),
        Exponential => (6.5,),
        FDist => (6.0, 8.0),
        FDist => (8.0, 6.0),
        FDist => (30, 40),
        Frechet => (),
        Frechet => (0.5,),
        Frechet => (3.0,),
        Frechet => (20.0,),
        Frechet => (60.0,),
        Frechet => (0.5, 2.0),
        Frechet => (3.0, 2.0),
        Gamma => (),
        Gamma => (2.0,),
        # Gamma => (1.0, 1.0),
        # Gamma => (3.0, 1.0),
        Gamma => (3.0, 2.0),
        GeneralizedExtremeValue => (1.0, 1.0, 1.0),
        GeneralizedExtremeValue => (0.0, 1.0, 0.0),
        GeneralizedExtremeValue => (0.0, 1.0, 1.1),
        GeneralizedExtremeValue => (0.0, 1.0, 0.6),
        GeneralizedExtremeValue => (0.0, 1.0, 0.3),
        GeneralizedExtremeValue => (1.0, 1.0, -1.0),
        GeneralizedExtremeValue => (-1.0, 0.5, 0.6),
        # GeneralizedPareto => (),
        # GeneralizedPareto => (1.0, 1.0),
        # GeneralizedPareto => (0.1, 2.0),
        # GeneralizedPareto => (1.0, 1.0, 1.0),
        # GeneralizedPareto => (-1.5, 0.5, 2.0),
        Gumbel => (),
        Gumbel => (3.0,),
        Gumbel => (3.0, 5.0),
        Gumbel => (5.0, 3.0),
        InverseGamma => (),
        InverseGamma => (2.0,),
        InverseGamma => (1.0, 1.0),
        InverseGamma => (1.0, 2.0),
        InverseGamma => (2.0, 1.0),
        InverseGamma => (2.0, 3.0),
        InverseGaussian => (),
        InverseGaussian => (0.8,),
        InverseGaussian => (2.0,),
        InverseGaussian => (1.0, 1.0),
        InverseGaussian => (2.0, 1.5),
        InverseGaussian => (2.0, 7.0),
        Laplace => (),
        Laplace => (2.0,),
        Laplace => (0.0, 1.0),
        Laplace => (5.0, 1.0),
        Laplace => (5.0, 1.5),
        Levy => (),
        Levy => (2,),
        Levy => (2, 8),
        Levy => (3.0, 3),
        Logistic => (),
        Logistic => (2.0,),
        Logistic => (0.0, 1.0),
        Logistic => (5.0, 1.0),
        Logistic => (2.0, 1.5),
        Logistic => (5.0, 1.5),
        LogNormal => (),
        LogNormal => (1.0,),
        LogNormal => (0.0, 2.0),
        LogNormal => (1.0, 2.0),
        LogNormal => (3.0, 0.5),
        LogNormal => (3.0, 1.0),
        LogNormal => (3.0, 2.0),
        # NoncentralBeta => (2, 2, 0),
        # NoncentralBeta => (2, 6, 5),
        # NoncentralChisq => (2, 2),
        # NoncentralChisq => (2, 5),
        # NoncentralF => (2, 2, 2),
        # NoncentralF => (8, 10, 5),
        # NoncentralT => (2, 2),
        # NoncentralT => (10, 2),
        # NormalInverseGaussian => (1.7, 1.8, 1.2, 2.3),
        NormalCanon => (),
        NormalCanon => (0.0, 1.0),
        NormalCanon => (-1.0, 2.5),
        NormalCanon => (2.0, 0.8),
        Pareto => (),
        Pareto => (2.0,),
        Pareto => (2.0, 1.5),
        Pareto => (3.0, 2.0),
        Rayleigh => (),
        Rayleigh => (3.0,),
        Rayleigh => (8.0,),
        # StudentizedRange => (2.0, 2.0),
        # StudentizedRange => (5.0, 10.0),
        # StudentizedRange => (10.0, 5.0),
        SymTriangularDist => (),
        SymTriangularDist => (3.0,),
        SymTriangularDist => (3.0, 0.5),
        SymTriangularDist => (3.0, 2.0),
        SymTriangularDist => (10.0, 8.0),
        TDist => (1.2,),
        TDist => (5.0,),
        TDist => (28.0,),
        TriangularDist => (0, 5),
        TriangularDist => (-7, 2),
        TriangularDist => (-4, 14, 3),
        TriangularDist => (2, 2000, 500),
        TriangularDist => (1, 3, 2),
        # TruncatedNormal not a distribution => (0, 1, -2, 2),
        # TruncatedNormal not a distribution => (3, 10, 7, 8),
        # TruncatedNormal not a distribution => (27, 3, 0, Inf),
        # TruncatedNormal not a distribution => (-5, 1, -Inf, -10),
        # TruncatedNormal not a distribution => (1.8, 1.2, -Inf, 0),
        Uniform => (),
        Uniform => (0.0, 2.0),
        Uniform => (3.0, 17.0),
        Uniform => (3.0, 3.1),
        # VonMises => (),
        # VonMises => (4.0,),
        # VonMises => (1.1, 2.5),
        Weibull => (),
        Weibull => (0.5,),
        Weibull => (5.0,),
        Weibull => (20.0, 1.0),
        Weibull => (1.0, 2.0),
        Weibull => (5.0, 2.0)
    ]
        test_types(Dist, Float64.(args))
    end
end

M = typeof(1.0u"m");
@testset "Support for continuous distributions supporting any eltype" begin
    @testset "Testing $Dist" for (Dist, args) in [
        Normal => (),
        Normal => (2.0,),
        Normal => (-3.0, 2.0),
        Normal => (1.0, 10.0),
    ]
        @testset "Testing $T for $Dist" for T in [Float64, Float16, Dual, M]
            test_types(Dist, T.(args))
        end
    end
end

@testset "Support for discrete distributions with only integer arguments" begin
    @testset "Testing $Dist" for (Dist, args) in [
        DiscreteUniform => (),
        DiscreteUniform => (6,),
        DiscreteUniform => (7,),
        DiscreteUniform => (0, 4),
        DiscreteUniform => (2, 8),
        Hypergeometric => (2, 2, 2),
        Hypergeometric => (3, 2, 2),
        Hypergeometric => (3, 2, 0),
        Hypergeometric => (3, 2, 5),
        Hypergeometric => (4, 5, 6),
        Hypergeometric => (60, 80, 100)
    ]
        @testset "Testing $T for $Dist" for T in [Int]
            test_types(Dist, T.(args))
        end
    end
end

@testset "Support for discrete distributions with non-integer arguments" begin
    @testset "Testing $Dist" for (Dist, args) in [
        # Bernoulli => (),
        # Bernoulli => (0.25,),
        # Bernoulli => (0.75,),
        # Bernoulli => (0.0,),
        # Bernoulli => (1.0,),
        BetaBinomial => (2, 0.2, 0.25),
        BetaBinomial => (10, 0.2, 0.25),
        BetaBinomial => (10, 2, 2.5),
        BetaBinomial => (10, 60, 40),
        Binomial => (),
        Binomial => (3,),
        Binomial => (5, 0.4),
        Binomial => (6, 0.8),
        Binomial => (100, 0.1),
        Binomial => (100, 0.9),
        Binomial => (10, 0.0),
        Binomial => (10, 1.0),
        # Geometric => (),
        # Geometric => (0.02,),
        # Geometric => (0.1,),
        # Geometric => (0.5,),
        # Geometric => (0.9,),
        # NegativeBinomial => (),
        # NegativeBinomial => (6,),
        # NegativeBinomial => (1, 0.5),
        # NegativeBinomial => (5, 0.6),
        # NegativeBinomial => (0.5, 0.5),
        # Poisson => (),
        # Poisson => (0.0,),
        # Poisson => (0.5,),
        # Poisson => (2.0,),
        # Poisson => (10.0,),
        # Poisson => (80.0,),
        # Skellam => (),
        # Skellam => (2.0,),
        # Skellam => (2.0, 3.0),
        # Skellam => (3.2, 1.8),
        FisherNoncentralHypergeometric => (8, 6, 10, 1),
        FisherNoncentralHypergeometric => (8, 6, 10, 10),
        FisherNoncentralHypergeometric => (8, 6, 10, 0.1),
        FisherNoncentralHypergeometric => (80, 60, 100, 1),
        FisherNoncentralHypergeometric => (80, 60, 100, 10),
        FisherNoncentralHypergeometric => (80, 60, 100, 0.1),
        WalleniusNoncentralHypergeometric => (8, 6, 10, 1),
        WalleniusNoncentralHypergeometric => (8, 6, 10, 10),
        WalleniusNoncentralHypergeometric => (8, 6, 10, 0.1),
        WalleniusNoncentralHypergeometric => (40, 30, 50, 1),
        WalleniusNoncentralHypergeometric => (40, 30, 50, 0.5),
        WalleniusNoncentralHypergeometric => (40, 30, 50, 2)
    ]
        test_types(Dist, args)
    end
end
