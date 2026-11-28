using Distributions
using ForwardDiff
using Test

using Distributions: expectation

@testset "Kumaraswamy" begin
    @testset "NaNs" begin
        D = Kumaraswamy(420, 69)
        for f in (pdf, logpdf, cdf, ccdf, logcdf, logccdf)
            @test isnan(f(D, NaN))
        end
    end
    @testset "skewness identities" begin
        @test skewness(Kumaraswamy(1, 1)) ≈ 0 atol=1e-12
        # Kumaraswamy(a, 1) = Beta(a, 1) and Kumaraswamy(1, b) = Beta(1, b)
        for shape in (0.2, 0.5, 0.99, 1.01, 2.0, 5.0)
            @test skewness(Kumaraswamy(shape, 1.0)) ≈ skewness(Beta(shape, 1.0)) rtol=1e-10
            @test skewness(Kumaraswamy(1.0, shape)) ≈ skewness(Beta(1.0, shape)) rtol=1e-10
        end
        @test skewness(Kumaraswamy(2, 2)) ≈ -32 / (77sqrt(11)) rtol=1e-10
    end
    @testset "$T" for T in (Float16, Float32, Float64, Int32, Int64, Rational{Int})
        D = Kumaraswamy(T(2), T(3))
        @test partype(D) === T
        @test typeof(@inferred rand(D)) === typeof(rand())
        tol = sqrt(eps(float(T)))
        @testset "gradlogpdf" begin
            for x in T(0):(T <: Integer ? one(T) : T(0.5)):T(20)
                # ForwardDiff returns 0 at x = 1, where `logpdf` is -Inf.
                fd = x == 1 ? -Inf : ForwardDiff.derivative(Base.Fix1(logpdf, D), x)
                gl = @inferred gradlogpdf(D, x)
                @test fd ≈ gl atol=tol
                if T <: AbstractFloat
                    @test gl isa T
                end
            end
        end
        @testset "median" begin
            m = @inferred median(D)
            @test m ≈ sqrt(1 - T(2)^(-1//3)) atol=tol
            if T <: AbstractFloat
                @test m isa T
            end
        end
        @testset "entropy" begin
            shannon = @inferred entropy(D)
            @test shannon ≈ (19//12 - log(T(6))) atol=tol
            if T <: AbstractFloat
                @test shannon isa T
            end
        end
        @testset "mode" begin
            m = @inferred mode(D)
            @test m ≈ inv(sqrt(T(5))) atol=tol
            if T <: AbstractFloat
                @test m isa T
            end
            @test isnan(mode(Kumaraswamy(1, 1)))
        end
        @testset "$f" for (f, n) in [(skewness, 3), (kurtosis, 4)]
            μ = mean(D)
            σ = std(D)
            y₁ = @inferred f(D)
            y₂ = expectation(x -> ((x - μ) / σ)^n, D) - 3 * (f === kurtosis)
            if T <: AbstractFloat
                @test y₁ isa T
            end
            @test y₁ ≈ y₂ atol=sqrt(tol) broken=(T === Float16 && f === kurtosis)
        end
    end
    @testset "limits" begin
        bathtub = Kumaraswamy(0.5, 0.5)
        @test logpdf(bathtub, 0) == logpdf(bathtub, 1) == Inf
        explike = Kumaraswamy(5, 1)
        @test logpdf(explike, 0) == -Inf
        @test logpdf(explike, 1) ≈ log(5)
        extreme = Kumaraswamy(typemax(Int), 3)
        @test gradlogpdf(extreme, 1) == -Inf
        @test gradlogpdf(extreme, 1//1) === (-1//0)
        @test isnan(gradlogpdf(Kumaraswamy(2, 1), 1//1))
    end
end
