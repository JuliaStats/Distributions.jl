using Distributions
using Distributions: ProductNamedTupleDistribution
using LinearAlgebra
using Random
using Test

@testset "ProductNamedTupleDistribution" begin
    @testset "Constructor" begin
        nt = (x=Normal(1.0, 2.0), y=Normal(3.0, 4.0))
        d = @inferred ProductNamedTupleDistribution(nt)
        @test d isa ProductNamedTupleDistribution
        @test d.dists === nt
        @test Distributions.variate_form(typeof(d)) === NamedTupleVariate{(:x, :y)}
        @test Distributions.value_support(typeof(d)) === Continuous

        nt = (
            x=Normal(),
            y=Dirichlet(10, 1.0),
            z=DiscreteUniform(1, 10),
            w=LKJCholesky(3, 2.0),
        )
        d = @inferred ProductNamedTupleDistribution(nt)
        @test d isa ProductNamedTupleDistribution
        @test d.dists === nt
        @test Distributions.variate_form(typeof(d)) === NamedTupleVariate{(:x, :y, :z, :w)}
        @test Distributions.value_support(typeof(d)) === Continuous
    end

    @testset "product_distribution" begin
        nt = (x=Normal(1.0, 2.0), y=Normal(3.0, 4.0))
        d = @inferred product_distribution(nt)
        @test d === ProductNamedTupleDistribution(nt)

        nt = (
            x=Normal(),
            y=Dirichlet(10, 1.0),
            z=DiscreteUniform(1, 10),
            w=LKJCholesky(3, 2.0),
        )
        d = @inferred product_distribution(nt)
        @test d === ProductNamedTupleDistribution(nt)
    end

    @testset "show" begin
        d = ProductNamedTupleDistribution((x=Gamma(1.0, 2.0), y=Normal()))
        @test sprint(show, d) == """
        ProductNamedTupleDistribution{(:x, :y)}(
        x: Gamma{Float64}(α=1.0, θ=2.0)
        y: Normal{Float64}(μ=0.0, σ=1.0)
        )
        """
    end

    @testset "Properties" begin
        @testset "eltype" begin
            nt = (x=Normal(1.0, 2.0), y=Normal(3.0, 4.0))
            d = ProductNamedTupleDistribution(nt)
            @test eltype(d) === Float64

            nt = (x=Normal(), y=Gamma())
            d = ProductNamedTupleDistribution(nt)
            @test eltype(d) === Float64

            nt = (x=Bernoulli(),)
            d = ProductNamedTupleDistribution(nt)
            @test eltype(d) === Bool

            nt = (x=Normal(), y=Bernoulli())
            d = ProductNamedTupleDistribution(nt)
            @test eltype(d) === Real

            nt = (w=LKJCholesky(3, 2.0),)
            d = ProductNamedTupleDistribution(nt)
            @test eltype(d) === LinearAlgebra.Cholesky{Float64,Array{Float64,2}}

            nt = (
                x=Normal(),
                y=Dirichlet(10, 1.0),
                z=DiscreteUniform(1, 10),
                w=LKJCholesky(3, 2.0),
            )
            d = ProductNamedTupleDistribution(nt)
            @test eltype(d) === Any
        end

        @testset "minimum" begin
            nt = (x=Normal(1.0, 2.0), y=Gamma(), z=MvNormal(ones(5)))
            d = ProductNamedTupleDistribution(nt)
            @test @inferred(minimum(d)) ==
                (x=minimum(nt.x), y=minimum(nt.y), z=minimum(nt.z))
        end

        @testset "maximum" begin
            nt = (x=Normal(1.0, 2.0), y=Gamma(), z=MvNormal(ones(5)))
            d = ProductNamedTupleDistribution(nt)
            @test @inferred(maximum(d)) ==
                (x=maximum(nt.x), y=maximum(nt.y), z=maximum(nt.z))
        end

        @testset "insupport" begin
            nt = (x=Normal(1.0, 2.0), y=Gamma(), z=Dirichlet(5, 1.0))
            d = ProductNamedTupleDistribution(nt)
            x = (x=rand(nt.x), y=rand(nt.y), z=rand(nt.z))
            @test @inferred(insupport(d, x))
            @test_throws MethodError insupport(d, NamedTuple{(:y, :z, :x)}(x))
            @test_throws MethodError insupport(d, NamedTuple{(:x, :y)}(x))
            @test !insupport(d, merge(x, (x=NaN,)))
            @test !insupport(d, merge(x, (y=-1,)))
            @test !insupport(d, merge(x, (z=fill(0.25, 4),)))
        end
    end

    @testset "Evaluation" begin
        nt = (x=Normal(1.0, 2.0), y=Gamma(), z=Dirichlet(5, 1.0), w=Bernoulli())
        d = ProductNamedTupleDistribution(nt)
        x = (x=rand(nt.x), y=rand(nt.y), z=rand(nt.z), w=rand(nt.w))
        @test @inferred(logpdf(d, x)) ==
            logpdf(nt.x, x.x) + logpdf(nt.y, x.y) + logpdf(nt.z, x.z) + logpdf(nt.w, x.w)
        @test @inferred(pdf(d, x)) == exp(logpdf(d, x))
        @test @inferred(loglikelihood(d, x)) == logpdf(d, x)
        xs = [(x=rand(nt.x), y=rand(nt.y), z=rand(nt.z), w=rand(nt.w)) for _ in 1:10]
        @test @inferred(loglikelihood(d, xs)) == sum(logpdf.(Ref(d), xs))
    end

    @testset "Statistics" begin
        nt = (x=Normal(1.0, 2.0), y=Gamma(), z=MvNormal(1.0:5.0), w=Poisson(100))
        d = ProductNamedTupleDistribution(nt)
        @test @inferred(mode(d)) == (x=mode(nt.x), y=mode(nt.y), z=mode(nt.z), w=mode(nt.w))
        @test @inferred(mean(d)) == (x=mean(nt.x), y=mean(nt.y), z=mean(nt.z), w=mean(nt.w))
        @test @inferred(var(d)) == (x=var(nt.x), y=var(nt.y), z=var(nt.z), w=var(nt.w))
        @test @inferred(entropy(d)) ==
            entropy(nt.x) + entropy(nt.y) + entropy(nt.z) + entropy(nt.w)

        d1 = ProductNamedTupleDistribution((x=Normal(1.0, 2.0), y=Gamma()))
        d2 = ProductNamedTupleDistribution((x=Normal(), y=Gamma(2.0, 3.0)))
        @test kldivergence(d1, d2) ==
            kldivergence(d1.dists.x, d2.dists.x) + kldivergence(d1.dists.y, d2.dists.y)

        d3 = ProductNamedTupleDistribution((x=Normal(1.0, 2.0), y=Gamma(6.0, 7.0)))
        @test std(d3) == (x=std(d3.dists.x), y=std(d3.dists.y))
    end

    @testset "Sampling" begin
        rng = MersenneTwister(973)

        @testset "rand" begin
            nt = (x=Normal(1.0, 2.0), y=Gamma(), z=Dirichlet(5, 1.0), w=Bernoulli())
            d = ProductNamedTupleDistribution(nt)
            rng = MersenneTwister(973)
            x1 = @inferred rand(rng, d)
            @test eltype(x1) === eltype(d)
            rng = MersenneTwister(973)
            x2 = (
                x=rand(rng, nt.x), y=rand(rng, nt.y), z=rand(rng, nt.z), w=rand(rng, nt.w)
            )
            @test x1 == x2
            x3 = rand(rng, d)
            @test x3 != x1

            xs1 = @inferred rand(rng, d, 10)
            @test length(xs1) == 10
            @test all(insupport.(Ref(d), xs1))

            xs2 = @inferred rand(rng, d, (2, 3, 4))
            @test size(xs2) == (2, 3, 4)
            @test all(insupport.(Ref(d), xs2))
        end

        @testset "rand!" begin
            d = ProductNamedTupleDistribution((
                x=Normal(1.0, 2.0), y=Gamma(), z=Dirichlet(5, 1.0), w=Bernoulli()
            ))
            x = rand(d)
            xs = Array{typeof(x)}(undef, (2, 3, 4))
            rand!(d, xs)
            @test all(insupport.(Ref(d), xs))
        end
    end
end
