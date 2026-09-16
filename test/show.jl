using Distributions
using LinearAlgebra
using PDMats: PDiagMat, PDMat, ScalMat
using Test

using Distributions: _showname, _showparams, namedparams

# a distribution that does not implement `namedparams`, as a downstream one might not
struct WithoutNamedparams <: ContinuousUnivariateDistribution
    a::Float64
    b::Float64
end

# the single-line forms are evaluated here, with only the packages that their parameters name
module RoundTrip
using Distributions
using FillArrays
using LinearAlgebra
using PDMats
end

@testset "show" begin
    dists = Any[
        Normal(),
        Normal(0.0f0, 1.0f0),
        Cauchy(),
        Poisson(3.0),
        Bernoulli(0.3),
        DiscreteUniform(1, 6),
        Skellam(1.0, 2.0),
        Categorical([0.2, 0.3, 0.5]),
        Categorical(fill(0.1, 10)),
        DiscreteNonParametric([1.0, 2.0, 3.0], [0.2, 0.3, 0.5]),
        Soliton(10, 5, 0.2),
        VonMises(1.0, 2.0),
        truncated(Normal(); lower=2.0),
        truncated(Normal(); upper=3.0),
        truncated(Normal(); lower=2.0, upper=3.0),
        censored(Normal(); lower=-1.0),
        censored(truncated(Normal(); lower=0.0); upper=2.0),
        2 * Beta(2.0, 3.0) + 1,
        OrderStatistic(Normal(), 5, 2),
        Distributions.EdgeworthMean(Normal(), 10),
        Dirichlet(ones(20)),
        JointOrderStatistics(Normal(), 5, (2, 3)),
        DirichletMultinomial(10, [2.0, 3.0]),
        VonMisesFisher([1.0, 0.0], 2.0),
        MvNormal(Diagonal(ones(2))),
        MvNormalCanon(ones(2)),
        MvLogNormal(MvNormal(zeros(2), Diagonal(ones(2)))),
        MvLogitNormal(MvNormal([1.0, 2.0], Diagonal([3.0, 4.0]))),
        MvLogitNormal(canonform(MvNormal([1.0, 2.0], Diagonal([3.0, 4.0])))),
        MvTDist(5.0, zeros(2), PDMat(Matrix(1.0I, 2, 2))),
        product_distribution([Normal(), Gamma(1.0, 2.0)]),
        product_distribution(Normal(), Gamma(1.0, 2.0)),
        product_distribution((x=Normal(), y=Dirichlet([2.0, 4.0]))),
        Wishart(5.0, PDMat(Matrix(1.0I, 3, 3))),
        InverseWishart(5.0, PDMat(Matrix(1.0I, 3, 3))),
        MatrixNormal(zeros(2, 3), PDMat(Matrix(1.0I, 2, 2)), PDMat(Matrix(1.0I, 3, 3))),
        MatrixBeta(3, 5.0, 5.0),
        MatrixFDist(5.0, 6.0, PDMat(Matrix(1.0I, 3, 3))),
        MatrixTDist(5.0, zeros(2, 3), PDMat(Matrix(1.0I, 2, 2)), PDMat(Matrix(1.0I, 3, 3))),
        LKJ(3, 1.0),
        LKJCholesky(5, 1.0),
        reshape(product_distribution([Normal(), Normal(), Normal(), Normal()]), 2, 2),
        MixtureModel([Normal(), Normal(2.0, 3.0)]),
        UnivariateGMM([0.0, 1.0], [1.0, 2.0], Categorical([0.4, 0.6])),
    ]
    samplers = Any[
        Distributions.AliasTable([0.2, 0.3, 0.5]),
        sampler(Normal()),
        sampler(Poisson(3.0)),
        sampler(Multinomial(5, [0.2, 0.3, 0.5])),
    ]

    firstline(d) = first(split(repr("text/plain", d), '\n'))

    @testset "single line: $(nameof(typeof(s)))" for s in [dists; samplers]
        @test !occursin('\n', repr(s))
        @test repr(s) == string(s) == "$s"
        @test !occursin('\n', repr((s, s)))  # also inside a container
    end

    @testset "round trip: $(nameof(typeof(d)))" for d in dists
        @test RoundTrip.eval(Meta.parse(repr(d))) == d
    end

    @testset "report: $(nameof(typeof(d)))" for d in dists
        report = repr("text/plain", d)
        @test !endswith(report, '\n')
        @test !occursin("\n\n", report)

        # the title is a name, not a call
        @test endswith(firstline(d), " distribution")
        @test !occursin('(', firstline(d))

        # every parameter is shown by its own `show` method, except where a distribution prints
        # something more informative than `namedparams`
        generic = which(_showparams, Tuple{IO,Distribution})
        if which(_showparams, Tuple{IO,typeof(d)}) === generic
            for (name, value) in pairs(namedparams(d))
                @test occursin("\n  $name ", report)
                @test occursin("= $(repr(value))", report)
            end
        end
    end

    @testset "interface" begin
        @test hasmethod(show, Tuple{IO,MIME"text/plain",Normal{Float64}})
        # the three-argument `show` is reserved for `MIME` types
        @test !hasmethod(show, Tuple{IO,Normal{Float64},Tuple{Symbol}})
        # `MvLogitNormal` used to define `show(io, d; indent)`
        @test all(methods(show)) do m
            m.module !== Distributions || isempty(Base.kwarg_decl(m))
        end

        # without `namedparams` the display falls back to the fields, `params` still throws
        d = WithoutNamedparams(1.0, 2.0)
        @test repr(d) == "WithoutNamedparams(1.0, 2.0)"
        @test repr("text/plain", d) == """
            WithoutNamedparams distribution
            Parameters:
              a = 1.0
              b = 2.0"""
        @test_throws MethodError namedparams(d)
        @test_throws MethodError params(d)
    end

    @testset "univariate" begin
        @test repr(Normal()) == "Normal(0.0, 1.0)"
        @test repr("text/plain", Normal()) == """
            Normal distribution
            Parameters:
              μ = 0.0
              σ = 1.0"""

        # the values disclose the type of the parameters
        @test repr(Normal(0.0f0, 1.0f0)) == "Normal(0.0f0, 1.0f0)"
        @test repr(Normal(1//2, 3//4)) == "Normal(1//2, 3//4)"
    end

    @testset "truncated and censored" begin
        d = truncated(Normal(); lower=0.0)
        @test repr(d) == "truncated(Normal(0.0, 1.0), 0.0, nothing)"
        @test repr("text/plain", d) == """
            Truncated Normal distribution
            Parameters:
              μ = 0.0
              σ = 1.0
            Truncation:
              lower = 0.0"""

        d = censored(Normal(); upper=2.0)
        @test repr(d) == "censored(Normal(0.0, 1.0), nothing, 2.0)"
        @test repr("text/plain", d) == """
            Censored Normal distribution
            Parameters:
              μ = 0.0
              σ = 1.0
            Censoring:
              upper = 2.0"""

        # every level of a nested wrapper contributes its own section
        d = censored(truncated(Normal(); lower=0.0); upper=2.0)
        @test repr(d) == "censored(truncated(Normal(0.0, 1.0), 0.0, nothing), nothing, 2.0)"
        @test repr("text/plain", d) == """
            Censored Truncated Normal distribution
            Parameters:
              μ = 0.0
              σ = 1.0
            Truncation:
              lower = 0.0
            Censoring:
              upper = 2.0"""
    end

    @testset "multivariate and matrix-variate" begin
        @test repr("text/plain", MvNormal(Diagonal(ones(2)))) == """
            Multivariate normal distribution
            Parameters:
              μ = Zeros(2)
              Σ = [1.0 0.0; 0.0 1.0]"""

        # the matrix parameter is rendered by its own `show` method
        d = Wishart(5.0, PDMat(Matrix(1.0I, 3, 3)))
        @test occursin("S  = $(repr(d.S))", repr("text/plain", d))
    end

    @testset "names" begin
        # a distribution that does not name itself is named by its type
        generic = which(_showname, Tuple{IO,Distribution})
        @testset "$(nameof(typeof(d)))" for d in dists
            which(_showname, Tuple{IO,typeof(d)}) === generic || continue
            @test firstline(d) == "$(nameof(typeof(d))) distribution"
        end

        # a distribution that names itself
        @testset "$(nameof(typeof(d)))" for (d, name) in [
            Categorical([0.2, 0.3, 0.5]) => "Categorical",
            MvNormal(ScalMat(2, 1.0)) => "Multivariate normal",
            MvNormal(Diagonal(ones(2))) => "Multivariate normal",
            MvNormal(ones(2), PDMat(Matrix(1.0I, 2, 2))) => "Multivariate normal",
            MvNormalCanon(ScalMat(2, 1.0)) => "Canonical multivariate normal",
            MvNormalCanon(ones(2), PDiagMat(ones(2))) => "Canonical multivariate normal",
            MvNormalCanon(ones(2), PDMat(Matrix(1.0I, 2, 2))) => "Canonical multivariate normal",
            MvLogNormal(MvNormal(zeros(2), Diagonal(ones(2)))) => "Multivariate log-normal",
            MvLogitNormal(MvNormal(ones(2), Diagonal(ones(2)))) => "Multivariate logit-normal",
            MvLogitNormal(canonform(MvNormal(ones(2), Diagonal(ones(2))))) =>
                "Canonical multivariate logit-normal",
            MvTDist(5.0, zeros(2), PDMat(Matrix(1.0I, 2, 2))) => "Multivariate Student's t",
            Distributions.GenericMvTDist(5.0, zeros(2), PDiagMat(ones(2))) =>
                "Multivariate Student's t",
        ]
            @test firstline(d) == "$name distribution"
        end
    end

    @testset "wrappers" begin
        # the parameters of the wrapped distribution are reported instead of the distribution
        # itself, in the parameterization it was constructed with
        d = MvLogitNormal(MvNormal(ones(2), Diagonal(ones(2))))
        @test repr("text/plain", d) == """
            Multivariate logit-normal distribution
            Parameters:
              μ = [1.0, 1.0]
              Σ = [1.0 0.0; 0.0 1.0]"""
        @test repr("text/plain", canonform(d)) == """
            Canonical multivariate logit-normal distribution
            Parameters:
              h = [1.0, 1.0]
              J = [1.0 0.0; 0.0 1.0]"""

        @test repr("text/plain", OrderStatistic(Normal(), 5, 2)) == """
            Order statistic of a Normal distribution
            Parameters:
              μ = 0.0
              σ = 1.0
            Order:
              n    = 5
              rank = 2"""

        @test repr("text/plain", JointOrderStatistics(Normal(), 5, (2, 3))) == """
            Joint order statistics of a Normal distribution
            Parameters:
              μ = 0.0
              σ = 1.0
            Order:
              n     = 5
              ranks = (2, 3)"""

        @test repr("text/plain", 2 * Beta(2.0, 3.0) + 1) == """
            Affine Beta distribution
            Parameters:
              α = 2.0
              β = 3.0
            Transformation:
              μ = 1
              σ = 2"""

        @test repr("text/plain", Distributions.EdgeworthMean(Normal(), 10)) == """
            Edgeworth expansion of the mean of a Normal distribution
            Parameters:
              μ = 0.0
              σ = 1.0
            Sample:
              n = 10.0"""

        d = MixtureModel([Normal(), Gamma(1.0, 2.0)], [0.25, 0.75])
        @test repr("text/plain", d) == """
            MixtureModel distribution
            Components:
              [1] Normal(0.0, 1.0)
              [2] Gamma(1.0, 2.0)
            Prior:
              Categorical([0.25, 0.75])"""

        @test repr("text/plain", reshape(Dirichlet([1.0, 2.0, 3.0, 4.0]), 2, 2)) == """
            Reshaped Dirichlet distribution
            Parameters:
              alpha = [1.0, 2.0, 3.0, 4.0]
            Size:
              2×2"""
    end

    @testset "`:limit` and `:compact`" begin
        # the number of components is limited only where `:limit` is set
        d = MixtureModel([Normal(i, 1.0) for i in 1:20])
        @test count("\n  [", repr("text/plain", d)) == 20
        @test !occursin("omitted", repr("text/plain", d))

        limited = repr("text/plain", d; context=:limit => true)
        @test count("\n  [", limited) == 8
        @test occursin("12 components omitted", limited)

        # `:limit` reaches the parameters themselves
        d = Dirichlet(ones(50))
        limited = repr(d; context=:limit => true)
        @test !occursin('\n', limited)
        @test occursin('…', limited)
        @test length(limited) < length(repr(d))

        d = Normal(1 / 3, 1 / 7)
        @test occursin(
            repr(1 / 3; context=:compact => true), repr(d; context=:compact => true)
        )
    end
end
