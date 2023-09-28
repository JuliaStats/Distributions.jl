# Tests on Multivariate Logit-Normal distributions
using Distributions
using ForwardDiff
using LinearAlgebra
using Random
using Test

####### Core testing procedure

function test_mvlogitnormal(d::MvLogitNormal; nsamples::Int=10^6)
    @test d.normal isa AbstractMvNormal
    dnorm = d.normal

    @testset "properties" begin
        @test length(d) == length(dnorm) + 1
        @test params(d) == params(dnorm)
        @test partype(d) == partype(dnorm)
        @test eltype(d) == eltype(dnorm)
        @test eltype(typeof(d)) == eltype(typeof(dnorm))
        @test location(d) == mean(dnorm)
        @test minimum(d) == fill(0, length(d))
        @test maximum(d) == fill(1, length(d))
        @test insupport(d, normalize(rand(length(d)), 1))
        @test !insupport(d, normalize(rand(length(d) + 1), 1))
        @test !insupport(d, rand(length(d)))
        x = rand(length(d) - 1)
        x = vcat(x, -sum(x))
        @test !insupport(d, x)
    end

    @testset "conversions" begin
        @test convert(typeof(d), d) === d
        T = partype(d) <: Float64 ? Float32 : Float64
        if dnorm isa MvNormal
            @test convert(MvLogitNormal{MvNormal{T}}, d).normal ==
                convert(MvNormal{T}, dnorm)
            @test partype(convert(MvLogitNormal{MvNormal{T}}, d)) <: T
            @test canonform(d) isa MvLogitNormal{<:MvNormalCanon}
            @test canonform(d).normal == canonform(dnorm)
        elseif dnorm isa MvNormalCanon
            @test convert(MvLogitNormal{MvNormalCanon{T}}, d).normal ==
                convert(MvNormalCanon{T}, dnorm)
            @test partype(convert(MvLogitNormal{MvNormalCanon{T}}, d)) <: T
            @test meanform(d) isa MvLogitNormal{<:MvNormal}
            @test meanform(d).normal == meanform(dnorm)
        end
    end

    @testset "sampling" begin
        X = rand(d, nsamples)
        Y = @views log.(X[1:(end - 1), :]) .- log.(X[end, :]')
        Ymean = vec(mean(Y; dims=2))
        Ycov = cov(Y; dims=2)
        for i in 1:(length(d) - 1)
            @test isapprox(
                Ymean[i], mean(dnorm)[i], atol=sqrt(var(dnorm)[i] / nsamples) * 8
            )
        end
        for i in 1:(length(d) - 1), j in 1:(length(d) - 1)
            @test isapprox(
                Ycov[i, j],
                cov(dnorm)[i, j],
                atol=sqrt(prod(var(dnorm)[[i, j]]) / nsamples) * 20,
            )
        end
    end

    @testset "fitting" begin
        X = rand(d, nsamples)
        dfit = fit_mle(MvLogitNormal, X)
        dfit_norm = dfit.normal
        for i in 1:(length(d) - 1)
            @test isapprox(
                mean(dfit_norm)[i], mean(dnorm)[i], atol=sqrt(var(dnorm)[i] / nsamples) * 8
            )
        end
        for i in 1:(length(d) - 1), j in 1:(length(d) - 1)
            @test isapprox(
                cov(dfit_norm)[i, j],
                cov(dnorm)[i, j],
                atol=sqrt(prod(var(dnorm)[[i, j]]) / nsamples) * 20,
            )
        end
        @test fit_mle(MvLogitNormal{IsoNormal}, X) isa MvLogitNormal{<:IsoNormal}
    end

    @testset "evaluation" begin
        X = rand(d, nsamples)
        for i in 1:min(100, nsamples)
            @test @inferred(logpdf(d, X[:, i])) ≈ log(pdf(d, X[:, i]))
            if dnorm isa MvNormal
                @test @inferred(gradlogpdf(d, X[:, i])) ≈
                    ForwardDiff.gradient(x -> logpdf(d, x), X[:, i])
            end
        end
        @test logpdf(d, X) ≈ log.(pdf(d, X))
        @test isequal(logpdf(d, zeros(length(d))), -Inf)
        @test isequal(logpdf(d, ones(length(d))), -Inf)
        @test isequal(pdf(d, zeros(length(d))), 0)
        @test isequal(pdf(d, ones(length(d))), 0)
    end
end

@testset "Results MvLogitNormal consistent with univariate LogitNormal" begin
    μ = randn()
    σ = rand()
    d = MvLogitNormal([μ], fill(σ^2, 1, 1))
    duni = LogitNormal(μ, σ)
    @test location(d) ≈ [location(duni)]
    x = normalize(rand(2), 1)
    @test logpdf(d, x) ≈ logpdf(duni, x[1])
    @test pdf(d, x) ≈ pdf(duni, x[1])
    @test (Random.seed!(9274); rand(d)[1]) ≈ (Random.seed!(9274); rand(duni))
end

###### General Testing

@testset "MvLogitNormal tests" begin
    mvnorm_params = [
        (randn(5), I * rand()),
        (randn(4), Diagonal(rand(4))),
        (Diagonal(rand(6)),),
        (randn(5), exp(Symmetric(randn(5, 5)))),
        (exp(Symmetric(randn(5, 5))),),
    ]
    @testset "wraps MvNormal" begin
        @testset "$(typeof(prms))" for prms in mvnorm_params
            d = MvLogitNormal(prms...)
            @test d == MvLogitNormal(MvNormal(prms...))
            test_mvlogitnormal(d; nsamples=10^4)
        end
    end
    @testset "wraps MvNormalCanon" begin
        @testset "$(typeof(prms))" for prms in mvnorm_params
            d = MvLogitNormal(MvNormalCanon(prms...))
            test_mvlogitnormal(d; nsamples=10^4)
        end
    end

    @testset "kldivergence" begin
        d1 = MvLogitNormal(randn(5), exp(Symmetric(randn(5, 5))))
        d2 = MvLogitNormal(randn(5), exp(Symmetric(randn(5, 5))))
        @test kldivergence(d1, d2) ≈ kldivergence(d1.normal, d2.normal)
    end

    VERSION ≥ v"1.8" && @testset "show" begin
        d = MvLogitNormal([1.0, 2.0, 3.0], Diagonal([4.0, 5.0, 6.0]))
        @test sprint(show, d) === """
        MvLogitNormal{DiagNormal}(
          DiagNormal(
          dim: 3
          μ: [1.0, 2.0, 3.0]
          Σ: [4.0 0.0 0.0; 0.0 5.0 0.0; 0.0 0.0 6.0]
          )
        )
        """
    end
end
