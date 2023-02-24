using Distributions
using FiniteDifferences
using ForwardDiff
using Random
using Test

using Base: Fix1
using Distributions: expectation

@testset "Lindley" begin
    @testset "NaNs" begin
        D = Lindley()
        for f in (pdf, logpdf, gradlogpdf, cdf, ccdf, logcdf, logccdf, mgf, cgf, cf)
            @test isnan(f(D, NaN))
        end
    end
    @testset "MLE" begin
        rng = MersenneTwister(420)
        samples = rand(rng, Lindley(1.5f0), 10_000)
        mle = fit_mle(Lindley, samples)
        @test mle isa Lindley
        @test shape(mle) ≈ 1.5 atol=0.1
    end
    @testset "$T" for T in (Float16, Float32, Float64, Rational{Int})
        D = Lindley(one(T))
        @test partype(D) === T
        @test typeof(@inferred rand(D)) === typeof(rand())
        @test @inferred(mean(D)) == T(3/2)
        tol = sqrt(eps(float(T)))
        @testset "Gradient of log PDF" begin
            for x in T(0):T(0.5):T(20)
                fd = ForwardDiff.derivative(Fix1(logpdf, D), x)
                gl = @inferred gradlogpdf(D, x)
                @test gl isa T
                @test fd ≈ gl atol=tol
            end
        end
        @testset "Entropy" begin
            shannon = @inferred entropy(D)
            expect = T(expectation(x -> -logpdf(D, x), D))
            if T <: AbstractFloat
                @test shannon isa T
            end
            @test shannon ≈ expect atol=tol
        end
        @testset "K-L divergence" begin
            S = supertype(typeof(D))
            D₂ = Lindley(T(2))
            d₁ = kldivergence(D, D₂)
            d₂ = invoke(kldivergence, Tuple{S,S}, D, D₂)
            if T <: AbstractFloat
                @test d₁ isa T
            end
            @test d₁ ≈ d₂ atol=tol
        end
        @testset "Mode" begin
            @test iszero(@inferred mode(D))
            @test isone(mode(Lindley(T(0.5))))
            m = mode(Lindley(T(0.1)))
            @test m isa T
            @test m ≈ T(9) atol=tol
        end
        @testset "Skewness" begin
            μ = mean(D)
            σ = std(D)
            s₁ = @inferred skewness(D)
            s₂ = T(expectation(x -> ((x - μ) / σ)^3, D))
            if T <: AbstractFloat
                @test s₁ isa T
            end
            @test s₁ ≈ s₂ atol=sqrt(tol)
        end
        @testset "MGF, CGF, CF" begin
            @test @inferred(mgf(D, 0)) === one(T)
            @test iszero(mgf(D, shape(D) + 1))
            @test ForwardDiff.derivative(Fix1(mgf, D), 0) ≈ mean(D)
            @test central_fdm(5, 1)(Fix1(cf, D), 0) ≈ mean(D) * im
            test_cgf(D, (-1e6, -100f0, Float16(-1), 1//10, 0.9))
        end
    end
end
