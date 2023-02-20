using Distributions
using FiniteDifferences

using Statistics
using Test

@testset "triangular" begin
    @testset "constructor" begin
        @test_throws ArgumentError TriangularDist(1, 0, 0)
        @test_throws ArgumentError TriangularDist(1, 1, 0)
        @test_throws ArgumentError TriangularDist(0, 0, 1)
    end

    @testset "type stability" begin
        Ts = (Float16, Float32, Float64, BigFloat, Rational{Int}, Int, BigInt)
        for T1 in Ts
            a = T1(1)
            b = T1(7)
            c = T1(5)
            dist = TriangularDist(a, b, c)
            distf64 = TriangularDist(Float64(a), Float64(b), Float64(c))

            for T2 in Ts
                xa = T2(1)
                xb = T2(7)
                xsmall = T2(0)
                xlarge = T2(8)
                x_middle_a_c = T2(3)
                x_middle_b_c = T2(6)
                t1 = T2(2)
                t2 = T2(5)

                for f in (pdf, cdf)
                    @inferred f(dist, xa) ≈ f(distf64, Float64(xa))
                    @inferred f(dist, xb) ≈ f(distf64, Float64(xb))
                    @inferred f(dist, xsmall) ≈ f(distf64, Float64(xsmall))
                    @inferred f(dist, xlarge) ≈ f(distf64, Float64(xlarge))
                    @inferred f(dist, x_middle_a_c) ≈ f(distf64, Float64(x_middle_a_c))
                    @inferred f(dist, x_middle_b_c) ≈ f(distf64, Float64(x_middle_b_c))
                end

                for f in (mgf, cf)
                    @inferred f(dist, t1) ≈ f(distf64, Float64(t1))
                    @inferred f(dist, t2) ≈ f(distf64, Float64(t2))
                end
            end
        end
    end

    @testset "interface" begin
        fdm = central_fdm(5, 1)
        fdm2 = central_fdm(5, 2)
        for (a, b, c) in ((2, 10, 8), (2, 6, 4), (2, 4, 4), (2, 4, 2), (2, 2, 2))
            d = TriangularDist(a, b, c)

            @test params(d) == (a, b, c)
            @test mode(d) == c
            @test mean(d) == (a + b + c) / 3
            @test median(d) == (c >= middle(a, b) ?
                                a + sqrt((b - a) * (c - a) / 2) :
                                b - sqrt((b - a) * (b - c) / 2))
            @test var(d) == (a^2 + b^2 + c^2 - a * b - a * c - b * c) / 18

            @test kurtosis(d) == -3 / 5
            @test entropy(d) == 1 / 2 + log((b - a) / 2)

            # x < a
            for x in (a - 1, a - 3)
                @test pdf(d, x) == 0
                @test logpdf(d, x) == -Inf
                @test cdf(d, x) == 0
            end
            # x = a
            @test pdf(d, a) == (a == b ? Inf : (a == c ? 2 / (b - a) : 0))
            @test logpdf(d, a) == log(pdf(d, a))
            @test cdf(d, a) == (a == b ? 1 : 0)
            # a < x < c
            if a < c
                x = (a + c) / 2
                @test pdf(d, x) == 2 * (x - a) / ((b - a) * (c - a))
                @test logpdf(d, x) == log(pdf(d, x))
                @test cdf(d, x) == (x - a)^2 / ((b - a) * (c - a))
            end
            # x = c
            @test pdf(d, c) == (a == b ? Inf : 2 / (b - a))
            @test logpdf(d, c) == log(pdf(d, c))
            @test cdf(d, c) == (c == b ? 1 : (c - a) / (b - a))
            # c < x < b
            if c < b
                x = (c + b) / 2
                @test pdf(d, x) == 2 * (b - x) / ((b - a) * (b - c))
                @test logpdf(d, x) == log(pdf(d, x))
                @test cdf(d, x) == 1 - (b - x)^2 / ((b - a) * (b - c))
            end
            # x = b
            @test pdf(d, b) == (b == a ? Inf : (b == c ? 2 / (b - a) : 0))
            @test logpdf(d, b) == log(pdf(d, b))
            @test cdf(d, b) == 1
            # x > b
            for x in (b + 1, b + 3)
                @test pdf(d, x) == 0
                @test logpdf(d, x) == -Inf
                @test cdf(d, x) == 1
            end

            @test quantile(d, 0) == a
            @test quantile(d, 0.5) == median(d)
            @test quantile(d, 1) == b

            @test mgf(d, 0) == 1
            @test fdm(Base.Fix1(mgf, d), 0.0) ≈ mean(d)
            @test fdm2(Base.Fix1(mgf, d), 0.0) ≈ mean(d)^2 + var(d) rtol=1e-6
            @test cf(d, 0) == 1
            @test fdm(Base.Fix1(cf, d), 0.0) ≈ mean(d) * im
            @test fdm2(Base.Fix1(cf, d), 0.0) ≈ -(mean(d)^2 + var(d)) rtol=1e-6
        end
    end
end
