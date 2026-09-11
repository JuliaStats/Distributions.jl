using Distributions
using LinearAlgebra
using PDMats: PDiagMat, PDMat, ScalMat
using Test

using Distributions: namedparams

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
        Dirichlet(ones(20)),
        DirichletMultinomial(10, [2.0, 3.0]),
        VonMisesFisher([1.0, 0.0], 2.0),
        MvNormal(Diagonal(ones(2))),
        MvNormalCanon(ones(2)),
        MvLogNormal(MvNormal(zeros(2), Diagonal(ones(2)))),
        MvLogitNormal(MvNormal([1.0, 2.0], Diagonal([3.0, 4.0]))),
        MvTDist(5.0, zeros(2), PDMat(Matrix(1.0I, 2, 2))),
        product_distribution([Normal(), Gamma(1.0, 2.0)]),
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
    lastline(d) = last(split(repr("text/plain", d), '\n'))

    @testset "single line: $(nameof(typeof(s)))" for s in [dists; samplers]
        @test !occursin('\n', repr(s))
        @test repr(s) == string(s) == "$s"
        @test !occursin('\n', repr((s, s)))  # also inside a container
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
        if !(d isa Union{
            Distributions.Censored,
            Distributions.ProductNamedTupleDistribution,
            MixtureModel,
            Truncated,
        })
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
    end

    @testset "univariate" begin
        @test repr(Normal()) == "Normal(0.0, 1.0)"
        @test repr("text/plain", Normal()) == """
            Normal distribution
            Parameters:
              μ = 0.0
              σ = 1.0
            Support:
              -Inf < x < Inf"""

        # the values disclose the type of the parameters
        @test repr(Normal(0.0f0, 1.0f0)) == "Normal(0.0f0, 1.0f0)"
        @test repr(Normal(1//2, 3//4)) == "Normal(1//2, 3//4)"
    end

    @testset "support" begin
        @test lastline(Normal()) == "  -Inf < x < Inf"
        @test lastline(Beta(2.0, 3.0)) == "  0.0 ≤ x ≤ 1.0"
        @test lastline(Exponential(1.5)) == "  0.0 ≤ x < Inf"
        @test lastline(Poisson(3.0)) == "  {0, 1, …}"
        @test lastline(Bernoulli(0.3)) == "  {false, true}"
        @test lastline(Binomial(10, 0.3)) == "  {0, 1, …, 10}"
        @test lastline(DiscreteUniform(1, 6)) == "  {1, 2, …, 6}"
        @test lastline(Skellam(1.0, 2.0)) == "  {…, -1, 0, 1, …}"
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
              lower = 0.0
            Support:
              0.0 ≤ x < Inf"""

        d = censored(Normal(); upper=2.0)
        @test repr(d) == "censored(Normal(0.0, 1.0), nothing, 2.0)"
        @test repr("text/plain", d) == """
            Censored Normal distribution
            Parameters:
              μ = 0.0
              σ = 1.0
            Censoring:
              upper = 2.0
            Support:
              -Inf < x ≤ 2.0"""

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
              upper = 2.0
            Support:
              0.0 ≤ x ≤ 2.0"""
    end

    @testset "multivariate and matrix-variate" begin
        @test repr("text/plain", MvNormal(Diagonal(ones(2)))) == """
            ZeroMeanDiagNormal distribution
            Parameters:
              μ = Zeros(2)
              Σ = [1.0 0.0; 0.0 1.0]
            Dimension:
              2"""

        # the matrix parameter is rendered by its own `show` method
        d = Wishart(5.0, PDMat(Matrix(1.0I, 3, 3)))
        @test occursin("S  = $(repr(d.S))", repr("text/plain", d))
        @test endswith(repr("text/plain", d), "Size:\n  3×3")
    end

    @testset "alias names" begin
        showname(d) = first(split(firstline(d), ' '))
        @test showname(MvNormal(ScalMat(2, 1.0))) == "ZeroMeanIsoNormal"
        @test showname(MvNormal(Diagonal(ones(2)))) == "ZeroMeanDiagNormal"
        @test showname(MvNormal(PDMat(Matrix(1.0I, 2, 2)))) == "ZeroMeanFullNormal"
        @test showname(MvNormal(ones(2), ScalMat(2, 1.0))) == "IsoNormal"
        @test showname(MvNormal(ones(2), Diagonal(ones(2)))) == "DiagNormal"
        @test showname(MvNormal(ones(2), PDMat(Matrix(1.0I, 2, 2)))) == "FullNormal"
        @test showname(MvNormalCanon(ScalMat(2, 1.0))) == "ZeroMeanIsoNormalCanon"
        @test showname(MvNormalCanon(PDiagMat(ones(2)))) == "ZeroMeanDiagNormalCanon"
        @test showname(MvNormalCanon(ones(2), ScalMat(2, 1.0))) == "IsoNormalCanon"
        @test showname(MvNormalCanon(ones(2), PDiagMat(ones(2)))) == "DiagNormalCanon"
        @test showname(MvNormalCanon(ones(2), PDMat(Matrix(1.0I, 2, 2)))) == "FullNormalCanon"
        @test showname(MvLogitNormal(MvNormal(ones(2), Diagonal(ones(2))))) ==
            "MvLogitNormal{DiagNormal}"
    end

    @testset "`:limit` and `:compact`" begin
        # only the REPL limits the number of components
        d = MixtureModel([Normal(i, 1.0) for i in 1:20])
        @test count("prior", repr("text/plain", d)) == 20
        @test !occursin("omitted", repr("text/plain", d))

        limited = repr("text/plain", d; context=:limit => true)
        @test count("prior", limited) == 8
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
