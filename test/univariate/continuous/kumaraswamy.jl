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
    @testset "$T" for T in (Float16, Float32, Float64, Int32, Int64, Rational{Int})
        D = Kumaraswamy(T(2), T(3))
        @test partype(D) === T
        @test typeof(@inferred rand(D)) === typeof(rand())
        tol = sqrt(eps(float(T)))
        @testset "gradlogpdf" begin
            for x in T(0):(T <: Integer ? one(T) : T(0.5)):T(20)
                fd = ForwardDiff.derivative(Base.Fix1(logpdf, D), x)
                gl = @inferred gradlogpdf(D, x)
                @test fd ≈ gl atol=tol
                if T <: AbstractFloat
                    @test gl isa T
                end
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
            @test y₁ ≈ y₂ atol=sqrt(tol)
        end
    end
end
