using Distributions
using InteractiveUtils
using LinearAlgebra
using PDMats: PDMat, ScalMat
using Test

using Distributions: namedparams

@testset "namedparams" begin
    dists = Any[
        # univariate continuous
        Arcsine(),
        Beta(2.0, 3.0),
        BetaPrime(),
        Biweight(0.0, 1.0),
        Cauchy(),
        Chernoff(),
        Chi(3.0),
        Chisq(3.0),
        Cosine(0.0, 1.0),
        Epanechnikov(0.0, 1.0),
        Erlang(3, 2.0),
        Exponential(1.5),
        FDist(3.0, 5.0),
        Frechet(),
        Gamma(1.0, 2.0),
        GeneralizedExtremeValue(0.0, 1.0, 0.1),
        GeneralizedPareto(0.0, 1.0, 0.1),
        Gumbel(),
        InverseGamma(),
        InverseGaussian(),
        JohnsonSU(0.0, 1.0, 0.0, 1.0),
        Kolmogorov(),
        Kumaraswamy(2.0, 3.0),
        Laplace(),
        Levy(),
        Lindley(1.5),
        LogNormal(),
        LogUniform(1.0, 2.0),
        LogitNormal(),
        LogLogistic(1.0, 2.0),
        Logistic(),
        Normal(),
        Normal(0.0f0, 1.0f0),
        Normal(1//2, 3//4),
        NormalCanon(),
        NormalInverseGaussian(0.0, 1.0, 0.5, 0.5),
        NoncentralBeta(2.0, 3.0, 1.0),
        NoncentralChisq(3.0, 1.0),
        NoncentralF(3.0, 5.0, 1.0),
        NoncentralT(5.0, 1.0),
        PGeneralizedGaussian(1.0),
        Pareto(1.0, 2.0),
        Rayleigh(),
        Rician(0.0, 1.0),
        Semicircle(2.0),
        SkewNormal(0.0, 1.0, 2.0),
        SkewedExponentialPower(),
        StudentizedRange(5.0, 3.0),
        SymTriangularDist(),
        TDist(5.0),
        TriangularDist(0.0, 1.0, 0.5),
        Triweight(0.0, 1.0),
        Uniform(0.0, 1.0),
        VonMises(1.0, 2.0),
        Weibull(1.0, 2.0),
        Distributions.KSDist(5),
        Distributions.KSOneSided(5),
        # univariate discrete
        Bernoulli(0.3),
        BernoulliLogit(0.5),
        BetaBinomial(10, 2.0, 3.0),
        Binomial(10, 0.3),
        Categorical([0.2, 0.3, 0.5]),
        Dirac(2.5),
        DiscreteNonParametric([1.0, 2.0, 3.0], [0.2, 0.3, 0.5]),
        DiscreteUniform(1, 6),
        FisherNoncentralHypergeometric(5, 5, 3, 2.0),
        Geometric(0.2),
        Hypergeometric(5, 5, 3),
        NegativeBinomial(5.0, 0.3),
        Poisson(3.0),
        PoissonBinomial([0.2, 0.5, 0.8]),
        Skellam(1.0, 2.0),
        Soliton(10, 5, 0.2),
        WalleniusNoncentralHypergeometric(5, 5, 3, 2.0),
        # wrappers and derived distributions
        truncated(Normal(); lower=2.0),
        truncated(Normal(); upper=3.0),
        truncated(Normal(); lower=2.0, upper=3.0),
        censored(Normal(); lower=-1.0),
        censored(Normal(); upper=2.0),
        censored(Normal(); lower=-1.0, upper=2.0),
        2 * Beta(2.0, 3.0) + 1,
        OrderStatistic(Normal(), 5, 2),
        Distributions.EdgeworthMean(Normal(), 10),
        Distributions.EdgeworthSum(Normal(), 10),
        Distributions.EdgeworthZ(Normal(), 10),
        # multivariate
        Dirichlet([2.0, 4.0]),
        DirichletMultinomial(10, [2.0, 3.0]),
        JointOrderStatistics(Normal(), 5, (2, 3)),
        Multinomial(5, [0.2, 0.3, 0.5]),
        MvLogNormal(MvNormal(zeros(2), Diagonal(ones(2)))),
        MvLogitNormal(MvNormal([1.0, 2.0], Diagonal([3.0, 4.0]))),
        MvNormal(Diagonal(ones(2))),
        MvNormal(zeros(2), ScalMat(2, 1.0)),
        MvNormal(ones(2), PDMat(Matrix(1.0I, 2, 2))),
        MvNormalCanon(ones(2)),
        MvNormalCanon(zeros(2), ones(2), PDMat(Matrix(1.0I, 2, 2))),
        MvTDist(5.0, zeros(2), PDMat(Matrix(1.0I, 2, 2))),
        VonMisesFisher([1.0, 0.0], 2.0),
        product_distribution(Normal(), Gamma(1.0, 2.0)),
        product_distribution([Normal(), Gamma(1.0, 2.0)]),
        product_distribution((x=Normal(), y=Dirichlet([2.0, 4.0]))),
        # matrix-variate
        InverseWishart(5.0, PDMat(Matrix(1.0I, 3, 3))),
        LKJ(3, 1.0),
        LKJCholesky(5, 1.0),
        MatrixBeta(3, 5.0, 5.0),
        MatrixFDist(5.0, 6.0, PDMat(Matrix(1.0I, 3, 3))),
        MatrixNormal(zeros(2, 3), PDMat(Matrix(1.0I, 2, 2)), PDMat(Matrix(1.0I, 3, 3))),
        MatrixTDist(5.0, zeros(2, 3), PDMat(Matrix(1.0I, 2, 2)), PDMat(Matrix(1.0I, 3, 3))),
        Wishart(5.0, PDMat(Matrix(1.0I, 3, 3))),
        reshape(product_distribution([Normal(), Normal(), Normal(), Normal()]), 2, 2),
        # mixtures
        MixtureModel([Normal(), Normal(2.0, 3.0)]),
        UnivariateGMM([0.0, 1.0], [1.0, 2.0], Categorical([0.4, 0.6])),
    ]

    # There is no fallback for `namedparams`, so a distribution missing from `dists` is not
    # covered by the tests below - and would be a `MethodError` in `params`.
    @testset "all distributions are covered" begin
        types = Any[Distribution]
        for T in types
            # the entries appended here are visited as well
            append!(types, subtypes(T))
        end
        @testset "$T" for T in types
            # other test files define distributions of their own
            parentmodule(T) === Distributions || continue
            isabstracttype(T) || @test any(d -> d isa T, dists)
        end
    end

    # the expected `params`: the values of `namedparams`, with wrapped distributions flattened
    flatten(x) = (x,)
    flatten(d::Distribution) = params(d)
    flattenall(nt::NamedTuple) = mapreduce(flatten, (x, y) -> (x..., y...), values(nt); init=())

    @testset "$(nameof(typeof(d)))" for d in dists
        nt = @inferred namedparams(d)
        @test nt isa NamedTuple
        @inferred params(d)

        if d isa Union{
            Distributions.AffineDistribution,Distributions.ReshapedDistribution,UnivariateGMM
        }
            # `params` reports the wrapped distribution instead of flattening it
            @test params(d) === values(nt)
        elseif d isa MixtureModel
            # `params` reports the parameters of the components
            @test params(d) == ([params(c) for c in d.components], probs(d))
        else
            @test params(d) == flattenall(nt)
        end
    end

    @testset "flattening of nested wrappers" begin
        # every level contributes its own parameters, also if the names repeat
        d = censored(truncated(Normal(); lower=0.0); upper=2.0)
        @test namedparams(d) == (uncensored=d.uncensored, lower=nothing, upper=2.0)
        @test params(d) === (0.0, 1.0, 0.0, nothing, nothing, 2.0)
    end

    @testset "reconstruction" begin
        # `namedparams` reports the arguments that the constructor of `d` expects
        @testset "$f" for (f, d) in [
            Normal => Normal(1.0, 2.0),
            Categorical => Categorical([0.2, 0.3, 0.5]),
            truncated => truncated(Normal(); lower=0.0),
            censored => censored(Normal(); upper=2.0),
            OrderStatistic => OrderStatistic(Normal(), 5, 2),
            MvNormal => MvNormal(ones(2), PDMat(Matrix(1.0I, 2, 2))),
            Wishart => Wishart(5.0, PDMat(Matrix(1.0I, 3, 3))),
            MixtureModel => MixtureModel([Normal(), Normal(2.0, 3.0)]),
            UnivariateGMM => UnivariateGMM([0.0, 1.0], [1.0, 2.0], Categorical([0.4, 0.6])),
        ]
            @test f(namedparams(d)...) == d
        end
    end
end
