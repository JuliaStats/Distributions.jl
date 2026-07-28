using Test
using Distributions, Random

rng = MersenneTwister(123)

@testset "Truncated normal, mean and variance" begin
    @test mean(truncated(Normal(0,1),100,115)) ≈ 100.00999800099926070518490239457545847490332879043
    @test mean(truncated(Normal(-2,3),50,70)) ≈ 50.171943499898757645751683644632860837133138152489
    @test mean(truncated(Normal(0,2),-100,0)) ≈ -1.59576912160573071175978423973752747390343452465973
    @test mean(truncated(Normal(0,1))) == 0
    @test mean(truncated(Normal(0,1); lower=-Inf, upper=Inf)) == 0
    @test mean(truncated(Normal(0,1); lower=0)) ≈ +√(2/π)
    @test mean(truncated(Normal(0,1); lower=0, upper=Inf)) ≈ +√(2/π)
    @test mean(truncated(Normal(0,1); upper=0)) ≈ -√(2/π)
    @test mean(truncated(Normal(0,1); lower=-Inf, upper=0)) ≈ -√(2/π)
    @test var(truncated(Normal(0,1),50,70)) ≈ 0.00039904318680389954790992722653605933053648912703600
    @test var(truncated(Normal(-2,3); lower=50, upper=70)) ≈ 0.029373438107168350377591231295634273607812172191712
    @test var(truncated(Normal(0,1))) == 1
    @test var(truncated(Normal(0,1); lower=-Inf, upper=Inf)) == 1
    @test var(truncated(Normal(0,1); lower=0)) ≈ 1 - 2/π
    @test var(truncated(Normal(0,1); lower=0, upper=Inf)) ≈ 1 - 2/π
    @test var(truncated(Normal(0,1); upper=0)) ≈ 1 - 2/π
    @test var(truncated(Normal(0,1); lower=-Inf, upper=0)) ≈ 1 - 2/π
    # https://github.com/JuliaStats/Distributions.jl/issues/827
    @test mean(truncated(Normal(1000000,1),0,1000)) ≈ 999.999998998998999001005011019018990904720462367106
    @test var(truncated(Normal(),999000,1e6)) ≥ 0
    @test var(truncated(Normal(1000000,1),0,1000)) ≥ 0
    # https://github.com/JuliaStats/Distributions.jl/issues/624
    @test rand(truncated(Normal(+Inf, 1), 0, 1)) ≈ 1
    @test rand(truncated(Normal(-Inf, 1), 0, 1)) ≈ 0
    # Type stability
    for T in (Float32, Float64)
        t = truncated(Normal(T(1.5), T(4.1)), 0, 1)
        m = @inferred mode(t)
        μ = @inferred mean(t)
        σ = @inferred std(t)
        @test m === T(1)
        @test μ ≈ 0.50494725270783081889610661619986770485973643194141
        @test μ isa T
        @test σ ≈ 0.28836356398830993140576947440881738258157196701554
        @test σ isa T
    end
end
@testset "Truncated normal $trunc" begin
    trunc = truncated(Normal(0, 1), -2, 2)
    @testset "Truncated normal $trunc with $func" for func in
        [(r = rand, r! = rand!),
         (r = ((d, n) -> rand(rng, d, n)), r! = ((d, X) -> rand!(rng, d, X)))]
        repeats = 1000000
        
        @test abs(mean(func.r(trunc, repeats))) < 0.01
        @test abs(median(func.r(trunc, repeats))) < 0.01
        @test abs(var(func.r(trunc, repeats)) - var(trunc)) < 0.01

        X = Matrix{Float64}(undef, 1000, 1000)
        func.r!(trunc, X)
        @test abs(mean(X)) < 0.01
        @test abs(median(X)) < 0.01
        @test abs(var(X) - var(trunc)) < 0.01
    end
end

@testset "Truncated normal should be numerically stable at low probability regions" begin
    original = Normal(-5.0, 0.2)
    trunc = truncated(original, 0.0, 5.0)
    for x in LinRange(0.0, 5.0, 100)
        @test isfinite(logpdf(original, x))
        @test isfinite(logpdf(trunc, x))
        @test isfinite(pdf(trunc, x))
    end
end

@testset "Truncated normal MGF" begin
    two = big(2)
    sqrt2 = sqrt(two)
    invsqrt2 = inv(sqrt2)
    inv2sqrt2 = inv(two * sqrt2)
    twoerfsqrt2 = two * erf(sqrt2)

    for T in (Float32, Float64, BigFloat)
        d = truncated(Normal{T}(zero(T), one(T)), -2, 2)
        @test @inferred(mgf(d, 0)) == 1
        @test @inferred(mgf(d, 1)) ≈ @bigly sqrt(ℯ) * (erf(invsqrt2) + erf(3 * invsqrt2)) / twoerfsqrt2
        @test @inferred(mgf(d, 2.5)) ≈ @bigly exp(25//8) * (erf(9 * inv2sqrt2) - erf(inv2sqrt2)) / twoerfsqrt2
    end

    d = truncated(Normal(3, 10), 7, 8)
    @test mgf(d, 0) == 1
    @test mgf(d, 1) == 0

    d = truncated(Normal(27, 3); lower=0)
    @test mgf(d, 0) == 1
    @test mgf(d, 1) ≈ @bigly 2 * exp(63//2) / (1 + erf(9 * invsqrt2))
    @test mgf(d, 2.5) ≈ @bigly 2 * exp(765//8) / (1 + erf(9 * invsqrt2))

    d = truncated(Normal(-5, 1); upper=-10)
    @test mgf(d, 0) == 1
    @test mgf(d, 1) ≈ @bigly erfc(3 * sqrt2) / (exp(9//2) * erfc(5 * invsqrt2))

    @test isnan(mgf(truncated(Normal(); upper=NaN), 0))

    @test mgf(truncated(Normal(), -Inf, Inf), 1) == mgf(Normal(), 1)

    @test mgf(truncated(Normal(), 2, 2), 1) == exp(2)
end
