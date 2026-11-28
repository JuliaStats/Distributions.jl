using Distributions
using ConstructionBase: constructorof, getfields, getproperties, setproperties
using ForwardDiff
using LinearAlgebra
using Test

# Generic tests for all distributions are in `test_constructionbase` (see testutils.jl)

struct NoNamedParams <: ContinuousUnivariateDistribution
    a::Float64
    b::Float64
end

@testset "ConstructionBase" begin
    @testset "setproperties (#2119)" begin
        Σ = [2.0 0.5; 0.5 1.0]
        @testset "$(nameof(typeof(d)))" for (d, patch, expected) in [
            (Normal(0.4, 1.3), (; μ = 1.0), Normal(1.0, 1.3)),
            (InverseGamma(2.5, 1.3), (; θ = 1.0), InverseGamma(2.5, 1.0)),
            (InverseGamma(2.5, 1.3), (; α = 1.0), InverseGamma(1.0, 1.3)),
            (NormalCanon(0.4, 2.0), (; η = 1.0), NormalCanon(1.0, 2.0)),
            (NormalInverseGaussian(0.4, 2.5, 1.0, 1.3), (; μ = 1.0), NormalInverseGaussian(1.0, 2.5, 1.0, 1.3)),
            (DiscreteUniform(1, 6), (; b = 4), DiscreteUniform(1, 4)),
            (Categorical([0.2, 0.8]), (; p = [0.5, 0.5]), Categorical([0.5, 0.5])),
            (PoissonBinomial([0.2, 0.4]), (; p = [0.3, 0.5]), PoissonBinomial([0.3, 0.5])),
            (Soliton(10, 3, 0.05, 1e-6), (; δ = 0.1), Soliton(10, 3, 0.1, 1e-6)),
            (MvTDist(3.0, [1.0, 2.0], Σ), (; df = 5.0), MvTDist(5.0, [1.0, 2.0], Σ)),
            (Dirichlet([1.0, 2.0]), (; alpha = [2.0, 3.0]), Dirichlet([2.0, 3.0])),
            (DirichletMultinomial(5, [1.0, 2.0]), (; α = [2.0, 3.0]), DirichletMultinomial(5, [2.0, 3.0])),
            (VonMisesFisher([0.6, 0.8], 2.0), (; κ = 3.0), VonMisesFisher([0.6, 0.8], 3.0)),
            (MvLogNormal([1.0, 2.0], Σ), (; μ = [0.0, 1.0]), MvLogNormal([0.0, 1.0], Σ)),
            # fields of these types include derived quantities that previously were not updated
            (truncated(Normal(), -1, 1), (; lower = 0.0), truncated(Normal(), 0.0, 1)),
            (truncated(Normal(); lower = -1.0), (; lower = 0.0), truncated(Normal(); lower = 0.0)),
            (truncated(Normal(); upper = 1.0), (; lower = 0.0), truncated(Normal(), 0.0, 1.0)),
            (censored(Normal(), -1.0, 1.0), (; lower = big(0.0)), censored(Normal(), big(0.0), big(1.0))),
            (censored(Normal(); upper = 1.0), (; upper = 2.0), censored(Normal(); upper = 2.0)),
            (VonMises(0.5, 2.0), (; κ = 3.0), VonMises(0.5, 3.0)),
            (Wishart(3.0, Σ), (; df = 4.0), Wishart(4.0, Σ)),
            (InverseWishart(3.0, Σ), (; df = 4.0), InverseWishart(4.0, Σ)),
            (LKJ(3, 1.5), (; η = 2.0), LKJ(3, 2.0)),
            (LKJCholesky(3, 1.5), (; η = 2.0), LKJCholesky(3, 2.0)),
            (MatrixNormal(zeros(2, 2), Σ, Σ), (; M = ones(2, 2)), MatrixNormal(ones(2, 2), Σ, Σ)),
        ]
            d_new = setproperties(d, patch)
            @test d_new isa typeof(expected)
            @test d_new == expected
            for (k, v) in pairs(patch)
                @test getproperties(d_new)[k] == v
            end
        end
    end

    @testset "derived fields can't be set" begin
        @test_throws "(:μ,) are not properties of NormalCanon; its properties are (:η, :λ)" setproperties(NormalCanon(), (; μ = 1.0))
        @test_throws "(:invd,) are not properties of InverseGamma; its properties are (:α, :θ)" setproperties(InverseGamma(), (; invd = Gamma()))
        @test_throws "(:loglcdf, :tp) are not properties of Truncated" setproperties(truncated(Normal(), -1, 1), (; loglcdf = 0.0, tp = 1.0))
    end

    @testset "`constructorof` accepts the fields" begin
        # not covered by `test_constructionbase`
        for d in (truncated(Normal(), -1, 1), truncated(Normal(); lower = 0.0), censored(Normal(), -1.0, 1.0))
            @test constructorof(typeof(d))(getfields(d)...) == d
        end
        # derived fields must be consistent
        @test_throws "inconsistent fields of InverseGamma" constructorof(InverseGamma)(Gamma(2.5, 1.0), 2.0)
        @test_throws "inconsistent fields of Dirichlet" constructorof(Dirichlet)([1.0, 2.0], 4.0, 0.0)
        d = PoissonBinomial([0.2, 0.4])
        @test constructorof(typeof(d))([0.2, 0.4], nothing) == d
        @test constructorof(typeof(d))([0.2, 0.4], d.pmf) == d # `d.pmf` computes and caches the pmf
        @test_throws "inconsistent fields of PoissonBinomial" constructorof(typeof(d))([0.2, 0.4], [1.0, 0.0, 0.0])
    end

    @testset "map parameters" begin
        d = InverseGamma(ForwardDiff.Dual(2.5, 1.0), ForwardDiff.Dual(1.3, 0.0))
        d_new = setproperties(d, map(ForwardDiff.value, getproperties(d)))
        @test d_new isa InverseGamma{Float64}
        @test d_new == InverseGamma(2.5, 1.3)
    end

    @testset "fallback to fields without `namedparams`" begin
        d = NoNamedParams(1.0, 2.0)
        @test getproperties(d) == (; a = 1.0, b = 2.0)
        @test setproperties(d, (; b = 3.0)) == NoNamedParams(1.0, 3.0)
        @test_throws ArgumentError setproperties(d, (; c = 3.0))
        @test_throws "(:c,) are not properties of NoNamedParams; its properties are (:a, :b)" setproperties(d, (; c = 3.0))
    end
end
