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
        # These are exact Beta families, whose skewness uses an independent
        # closed formula. All reference skewnesses here are nonzero.
        for shape in (0.2, 0.5, 0.99, 1.01, 2.0, 5.0)
            @test skewness(Kumaraswamy(shape, 1.0)) ≈ skewness(Beta(shape, 1.0)) rtol=1e-10
            @test skewness(Kumaraswamy(1.0, shape)) ≈ skewness(Beta(1.0, shape)) rtol=1e-10
        end
        @test skewness(Kumaraswamy(2, 2)) ≈ -32 / (77sqrt(11)) rtol=1e-10
    end
    @testset "outside-support gradients" for T in (Float16, Float32, Float64, Int32, Int64, Rational{Int})
        d = Kumaraswamy(T(2), T(3))
        for x in (T(-20), T(-1), T(2), T(20))
            @test iszero(@inferred gradlogpdf(d, x))
            @test gradlogpdf(d, x) == gradlogpdf(Beta(T(2), T(3)), x)
        end
    end
    @testset "gradient limits and interior values" for T in (Float16, Float32, Float64)
        for (a, b) in ((0.5, 0.5), (1.0, 1.0), (1.0, 2.0), (2.0, 1.0), (2.0, 3.0), (3.0, 0.5))
            d = Kumaraswamy(T(a), T(b))
            left = a == 1 ? T(1 - b) : T(a > 1 ? Inf : -Inf)
            right = b == 1 ? T(a - 1) : T(b > 1 ? -Inf : Inf)
            @test @inferred(gradlogpdf(d, zero(T))) === left
            @test @inferred(gradlogpdf(d, -zero(T))) === left
            @test @inferred(gradlogpdf(d, one(T))) === right
            @test isnan(gradlogpdf(d, T(NaN)))
            for x in T.((0.125, 0.5, 0.875))
                reference = (a - 1) / x - a * (b - 1) * x^(a - 1) / (1 - x^a)
                @test gradlogpdf(d, x) ≈ reference rtol=sqrt(eps(T)) atol=sqrt(eps(T))
                # The interior arithmetic itself is unchanged.
                xa = x^T(a)
                old = (T(a) * (T(b) * xa - 1) + (1 - xa)) / (x * (xa - 1))
                @test gradlogpdf(d, x) === old
            end
        end
    end
    @testset "unsigned and wide parameter endpoint limits" begin
        # Subtract in the result type: 1 - UInt(3) wraps before conversion.
        for T in (UInt, UInt128, Int128, BigInt)
            left = @inferred gradlogpdf(Kumaraswamy(T(1), T(3)), 0.0)
            right = @inferred gradlogpdf(Kumaraswamy(T(3), T(1)), 1.0)
            @test left == -2
            @test right == 2
            @test iszero(@inferred gradlogpdf(Kumaraswamy(T(1), T(3)), -1.0))
            @test iszero(@inferred gradlogpdf(Kumaraswamy(T(3), T(1)), 2.0))
        end
        @test gradlogpdf(Kumaraswamy(UInt(1), UInt(3)), 0.0) === -2.0
        @test gradlogpdf(Kumaraswamy(UInt(3), UInt(1)), 1.0) === 2.0
        @test gradlogpdf(Kumaraswamy(UInt(1), typemax(UInt)), 0.0) == 1.0 - Float64(typemax(UInt))
        @test gradlogpdf(Kumaraswamy(typemax(UInt), UInt(1)), 1.0) == Float64(typemax(UInt)) - 1.0
    end
    @testset "kurtosis polynomial identity" for T in (Float16, Float32, Float64)
        # Density 6x(1-x²)^2 gives raw moments16/35,1/4,16/105,1/10.
        # This exact excess kurtosis uses no shared moment helper/quadrature.
        tolerance = T === Float16 ? 8eps(T) : T === Float32 ? 512eps(T) : 1e-10
        @test kurtosis(Kumaraswamy(T(2), T(3))) ≈ T(-88993 / 121203) rtol=tolerance
    end
    @testset "$T" for T in (Float16, Float32, Float64, Int32, Int64, Rational{Int})
        D = Kumaraswamy(T(2), T(3))
        @test partype(D) === T
        @test typeof(@inferred rand(D)) === typeof(rand())
        tol = sqrt(eps(float(T)))
        @testset "gradlogpdf" begin
            for x in T(0):(T <: Integer ? one(T) : T(0.5)):T(20)
                # logpdf clamps at the boundaries, where automatic
                # differentiation does not supply the one-sided limit.
                fd = iszero(x) ? Inf : x == 1 ? -Inf :
                     ForwardDiff.derivative(Base.Fix1(logpdf, D), x)
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
            # Integrate a widened density and its central moments rather than
            # sharing the low-precision beta-moment mean/variance with f(D).
            reference = convert(Kumaraswamy{Float64}, D)
            μ = expectation(identity, reference)
            σ = sqrt(expectation(x -> (x - μ)^2, reference))
            y₁ = @inferred f(D)
            y₂ = expectation(x -> ((x - μ) / σ)^n, reference) - 3 * (f === kurtosis)
            if T <: AbstractFloat
                @test y₁ isa T
            end
            @test y₁ ≈ y₂ atol=sqrt(tol)
        end
    end
    @testset "limits" begin
        bathtub = Kumaraswamy(0.5, 0.5)
        @test logpdf(bathtub, 0) == logpdf(bathtub, 1) == Inf
        explike = Kumaraswamy(5, 1)
        @test logpdf(explike, 0) == -Inf
        @test logpdf(explike, 1) ≈ log(5)
    end
end
