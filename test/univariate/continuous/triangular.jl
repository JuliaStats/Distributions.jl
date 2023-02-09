using Distributions
using Statistics
using Test

ϵ = .01

@testset "triangular" begin
    
    @test_throws ArgumentError TriangularDist(1, 0, 0)
    @test_throws ArgumentError TriangularDist(1, 1, 0)
    @test_throws ArgumentError TriangularDist(0, 0, 1)
    @test_throws ArgumentError TriangularDist(0, 0, 0)

    @testset "special_triangle" begin
        a = 0
        b = 1
        c = 1
        xa = a
        xb = b
        xlow = a - ϵ
        xhigh = b + ϵ
        xac = middle(a, c)
        xbc = middle(b, c)

        d = TriangularDist(a, b, c)
        @test mean(d) == 2 / 3
        @test var(d) == 1 / 18
        @test pdf(d, xa) == 2 * xa
        @test pdf(d, xb) == 2 * xb
        @test pdf(d, xac) == 2 * xac
        @test pdf(d, xbc) == 2 * xbc
        @test pdf(d, xlow) == 0
        @test pdf(d, xhigh) == 0
        @test cdf(d, xa) == xa^2
        @test cdf(d, xb) == xb^2
        @test cdf(d, xac) == xac^2
        @test cdf(d, xbc) == xbc^2
        @test cdf(d, xlow) == 0
        @test cdf(d, xhigh) == 1
    end

    @testset "regular_triangle" begin
        a = 1
        b = 5
        c = 4
        xa = a
        xb = b
        xlow = a - ϵ
        xhigh = b + ϵ
        xac = middle(a, c)
        xbc = middle(b, c)
        t1 = 1
        t2 = 2

        d = TriangularDist(a, b, c)
        @test params(d) == (a, b, c)
        @test mode(d) == c
        @test mean(d) == (a + b + c) / 3
        @test median(d) == (c >= middle(a, b) ?
            a + sqrt((b - a) * (c - a) / 2) :
            b - sqrt((b - a) * (b - c) / 2))
        @test var(d) == (a^2 + b^2 + c^2 - a * b - a * c - b * c) / 18
        @test skewness(d) == sqrt(2) * (a + b - 2c) * (2a - b - c) * (a - 2b + c) /
            (5 * (a^2 + b^2 + c^2 - a * b - a * c - b * c) ^ (3 / 2))
        @test kurtosis(d) == -3 / 5
        @test entropy(d) == 1 / 2 + log((b - a) / 2)
        @test pdf(d, xa) == 2 * (xa - a) / ((b - a) * (c - a))
        @test pdf(d, xb) == 2 * (b - xb) / ((b - a) * (b - c))
        @test pdf(d, xac) == 2 * (xac - a) / ((b - a) * (c - a))
        @test pdf(d, xbc) == 2 * (b - xbc) / ((b - a) * (b - c))
        @test pdf(d, xlow) == 0
        @test pdf(d, xhigh) == 0
        @test cdf(d, xa) == 0
        @test cdf(d, xb) == 1
        @test cdf(d, xac) == (xac - a)^2 / ((b - a) * (c - a))
        @test cdf(d, xbc) == 1 - (b - xbc)^2 / ((b - a) * (b - c))
        @test cdf(d, xlow) == 0
        @test cdf(d, xhigh) == 1
        @test quantile(d, 0) == xa
        @test quantile(d, 0.5) == median(d)
        @test quantile(d, 1) == xb
        @test isequal(mgf(d, 0), 1)
        @test isequal(mgf(d, t1), 2 * ((b - c) * exp(a * t1) - (b - a) * exp(c * t1) +
            (c - a) * exp(b * t1)) / ((b - a) * (c - a) * (b - c) * t1^2))
        @test isequal(mgf(d, t2), 2 * ((b - c) * exp(a * t2) - (b - a) * exp(c * t2) +
            (c - a) * exp(b * t2)) / ((b - a) * (c - a) * (b - c) * t2^2))
        @test isequal(cf(d, 0), complex(1))
        @test isequal(cf(d, t1), -2 * ((b - c) * cis(a * t1) - (b - a) * cis(c * t1) +
            (c - a) * cis(b * t1)) / ((b - a) * (c - a) * (b - c) * t1^2))
        @test isequal(cf(d, t2), -2 * ((b - c) * cis(a * t2) - (b - a) * cis(c * t2) +
            (c - a) * cis(b * t2)) / ((b - a) * (c - a) * (b - c) * t2^2))
    end

    @testset "symmetric_triangle" begin
        a = 1
        b = 3
        c = 2
        xa = a
        xb = b
        xlow = a - ϵ
        xhigh = b + ϵ
        xac = middle(a, c)
        xbc = middle(b, c)
        t1 = 1
        t2 = 2

        d = TriangularDist(a, b, c)
        @test params(d) == (a, b, c)
        @test mode(d) == c
        @test mean(d) == (a + b + c) / 3
        @test median(d) == (c >= middle(a, b) ?
            a + sqrt((b - a) * (c - a) / 2) :
            b - sqrt((b - a) * (b - c) / 2))
        @test var(d) == (a^2 + b^2 + c^2 - a * b - a * c - b * c) / 18
        @test skewness(d) == sqrt(2) * (a + b - 2c) * (2a - b - c) * (a - 2b + c) /
            (5 * (a^2 + b^2 + c^2 - a * b - a * c - b * c) ^ (3 / 2))
        @test kurtosis(d) == -3 / 5
        @test entropy(d) == 1 / 2 + log((b - a) / 2)
        @test pdf(d, xa) == 2 * (xa - a) / ((b - a) * (c - a))
        @test pdf(d, xb) == 2 * (b - xb) / ((b - a) * (b - c))
        @test pdf(d, xac) == 2 * (xac - a) / ((b - a) * (c - a))
        @test pdf(d, xbc) == 2 * (b - xbc) / ((b - a) * (b - c))
        @test pdf(d, xlow) == 0
        @test pdf(d, xhigh) == 0
        @test cdf(d, xa) == 0
        @test cdf(d, xb) == 1
        @test cdf(d, xac) == (xac - a)^2 / ((b - a) * (c - a))
        @test cdf(d, xbc) == 1 - (b - xbc)^2 / ((b - a) * (b - c))
        @test cdf(d, xlow) == 0
        @test cdf(d, xhigh) == 1
        @test quantile(d, 0) == xa
        @test quantile(d, 0.5) == median(d)
        @test quantile(d, 1) == xb
        @test isequal(mgf(d, 0), 1)
        @test isequal(mgf(d, t1), 2 * ((b - c) * exp(a * t1) - (b - a) * exp(c * t1) +
            (c - a) * exp(b * t1)) / ((b - a) * (c - a) * (b - c) * t1^2))
        @test isequal(mgf(d, t2), 2 * ((b - c) * exp(a * t2) - (b - a) * exp(c * t2) +
            (c - a) * exp(b * t2)) / ((b - a) * (c - a) * (b - c) * t2^2))
        @test isequal(cf(d, 0), complex(1))
        @test isequal(cf(d, t1), -2 * ((b - c) * cis(a * t1) - (b - a) * cis(c * t1) +
            (c - a) * cis(b * t1)) / ((b - a) * (c - a) * (b - c) * t1^2))
        @test isequal(cf(d, t2), -2 * ((b - c) * cis(a * t2) - (b - a) * cis(c * t2) +
            (c - a) * cis(b * t2)) / ((b - a) * (c - a) * (b - c) * t2^2))
    end

    @testset "right_triangle" begin
        a = 1
        b = 2
        c = 2
        xa = a
        xb = b
        xlow = a - ϵ
        xhigh = b + ϵ
        xac = middle(a, c)
        xbc = middle(b, c)
        t1 = 1
        t2 = 2

        d = TriangularDist(a, b, c)
        @test params(d) == (a, b, c)
        @test mode(d) == c
        @test mean(d) == (a + b + c) / 3
        @test median(d) == (c >= middle(a, b) ?
            a + sqrt((b - a) * (c - a) / 2) :
            b - sqrt((b - a) * (b - c) / 2))
        @test var(d) == (a^2 + b^2 + c^2 - a * b - a * c - b * c) / 18
        @test skewness(d) == sqrt(2) * (a + b - 2c) * (2a - b - c) * (a - 2b + c) /
            (5 * (a^2 + b^2 + c^2 - a * b - a * c - b * c) ^ (3 / 2))
        @test kurtosis(d) == -3 / 5
        @test entropy(d) == 1 / 2 + log((b - a) / 2)
        @test pdf(d, xa) == 2 * (xa - a) / ((b - a) * (c - a))
        @test pdf(d, xb) == 2 * (xb - a) / ((b - a) * (c - a))
        @test pdf(d, xac) == 2 * (xac - a) / ((b - a) * (c - a))
        @test pdf(d, xbc) == 2 * (xbc - a) / ((b - a) * (c - a))
        @test pdf(d, xlow) == 0
        @test pdf(d, xhigh) == 0
        @test cdf(d, xa) == 0
        @test cdf(d, xb) == 1
        @test cdf(d, xac) == (xac - a)^2 / ((b - a) * (c - a))
        @test cdf(d, xbc) == (xbc - a)^2 / ((b - a) * (c - a))
        @test cdf(d, xlow) == 0
        @test cdf(d, xhigh) == 1
        @test quantile(d, 0) == xa
        @test quantile(d, 0.5) == median(d)
        @test quantile(d, 1) == xb
        @test isequal(mgf(d, 0), 1)
        @test isequal(mgf(d, t1), 2 * ((b - c) * exp(a * t1) - (b - a) * exp(c * t1) +
            (c - a) * exp(b * t1)) / ((b - a) * (c - a) * (b - c) * t1^2))
        @test isequal(mgf(d, t2), 2 * ((b - c) * exp(a * t2) - (b - a) * exp(c * t2) +
            (c - a) * exp(b * t2)) / ((b - a) * (c - a) * (b - c) * t2^2))
        @test isequal(cf(d, 0), complex(1))
        @test isequal(cf(d, t1), -2 * ((b - c) * cis(a * t1) - (b - a) * cis(c * t1) +
            (c - a) * cis(b * t1)) / ((b - a) * (c - a) * (b - c) * t1^2))
        @test isequal(cf(d, t2), -2 * ((b - c) * cis(a * t2) - (b - a) * cis(c * t2) +
            (c - a) * cis(b * t2)) / ((b - a) * (c - a) * (b - c) * t2^2))
    end

    @testset "left_triangle" begin
        a = 1
        b = 2
        c = 1
        xa = a
        xb = b
        xlow = a - ϵ
        xhigh = b + ϵ
        xac = middle(a, c)
        xbc = middle(b, c)
        t1 = 1
        t2 = 2

        d = TriangularDist(a, b, c)
        @test params(d) == (a, b, c)
        @test mode(d) == c
        @test mean(d) == (a + b + c) / 3
        @test median(d) == (c >= middle(a, b) ?
            a + sqrt((b - a) * (c - a) / 2) :
            b - sqrt((b - a) * (b - c) / 2))
        @test var(d) == (a^2 + b^2 + c^2 - a * b - a * c - b * c) / 18
        @test skewness(d) == sqrt(2) * (a + b - 2c) * (2a - b - c) * (a - 2b + c) /
            (5 * (a^2 + b^2 + c^2 - a * b - a * c - b * c) ^ (3 / 2))
        @test kurtosis(d) == -3 / 5
        @test entropy(d) == 1 / 2 + log((b - a) / 2)
        @test pdf(d, xa) == 2 / (b - a)
        @test pdf(d, xb) == 2 * (b - xb) / ((b - a) * (b - c))
        @test pdf(d, xac) == 2 / (b - a)
        @test pdf(d, xbc) == 2 * (b - xbc) / ((b - a) * (b - c))
        @test pdf(d, xlow) == 0
        @test pdf(d, xhigh) == 0
        @test cdf(d, xa) == 0
        @test cdf(d, xb) == 1
        @test cdf(d, xac) == 0
        @test cdf(d, xbc) == 1 - (b - xbc)^2 / ((b - a) * (b - c))
        @test cdf(d, xlow) == 0
        @test cdf(d, xhigh) == 1
        @test quantile(d, 0) == xa
        @test quantile(d, 0.5) == median(d)
        @test quantile(d, 1) == xb
        @test isequal(mgf(d, 0), 1)
        @test isequal(mgf(d, t1), 2 * ((b - c) * exp(a * t1) - (b - a) * exp(c * t1) +
            (c - a) * exp(b * t1)) / ((b - a) * (c - a) * (b - c) * t1^2))
        @test isequal(mgf(d, t2), 2 * ((b - c) * exp(a * t2) - (b - a) * exp(c * t2) +
            (c - a) * exp(b * t2)) / ((b - a) * (c - a) * (b - c) * t2^2))
        @test isequal(cf(d, 0), complex(1))
        @test isequal(cf(d, t1), -2 * ((b - c) * cis(a * t1) - (b - a) * cis(c * t1) +
            (c - a) * cis(b * t1)) / ((b - a) * (c - a) * (b - c) * t1^2))
        @test isequal(cf(d, t2), -2 * ((b - c) * cis(a * t2) - (b - a) * cis(c * t2) +
            (c - a) * cis(b * t2)) / ((b - a) * (c - a) * (b - c) * t2^2))
    end

end
