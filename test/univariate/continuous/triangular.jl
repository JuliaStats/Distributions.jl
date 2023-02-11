using Distributions
using Statistics
using Test

ϵ = .01
ψ = 1

function convertstr(t::String, x::T) where T<:Real
    t == "Float16" && return convert(Float16, x)
    t == "Float32" && return convert(Float32, x)
    t == "Float64" && return convert(Float64, x)
    t == "BigFloat" && return convert(BigFloat, x)
    t == "Rational" && return convert(Rational, x)
    t == "Int8" && return convert(Int8, x)
    t == "Int16" && return convert(Int16, x)
    t == "Int32" && return convert(Int32, x)
    t == "Int64" && return convert(Int64, x)
    t == "Int128" && return convert(Int128, x)
    t == "BigInt" && return convert(BigInt, x)
    t == "UInt8" && return convert(UInt8, x)
    t == "UInt16" && return convert(UInt16, x)
    t == "UInt32" && return convert(UInt32, x)
    t == "UInt64" && return convert(UInt64, x)
    t == "UInt128" && return convert(UInt128, x)
    # t == "Bool" && return convert(Bool, x)
end

types = ["Float16", "Float32", "Float64", "BigFloat", "Rational"
        , "Int8", "Int16", "Int32", "Int64", "Int128", "BigInt"
        , "UInt8", "UInt16", "UInt32", "UInt64", "UInt128"] #Bool
n = length(types)^2
typeslist = Array{String}(undef, n, 2)
j = 1
for t in eachindex(types)
    for u in eachindex(types)
        typeslist[j, 1] = types[t]
        typeslist[j, 2] = types[u]
        global j += 1
    end
end

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

    @testset "type_stable_regular_triangle" begin
        a = 2
        b = 10
        c = 8
        xa = a
        xb = b
        xlow = a - ψ
        xhigh = b + ψ
        xac = middle(a, c)
        xbc = middle(b, c)
        t1 = 1
        t2 = 2

        for j = 1:n
            typep = typeslist[j, 1]
            typex = typeslist[j, 2]
            za = convertstr(typep, a)
            zb = convertstr(typep, b)
            zc = convertstr(typep, c)
            zxa = convertstr(typex, xa)
            zxb = convertstr(typex, xb)
            zxlow = convertstr(typex, xlow)
            zxhigh = convertstr(typex, xhigh)
            zxac = convertstr(typex, xac)
            zxbc = convertstr(typex, xbc)
            zt1 = convertstr(typex, t1)
            zt2 = convertstr(typex, t2)

            @inferred pdf(TriangularDist(za, zb, zc), zxa)
            @inferred pdf(TriangularDist(za, zb, zc), zxb)
            @inferred pdf(TriangularDist(za, zb, zc), zxlow)
            @inferred pdf(TriangularDist(za, zb, zc), zxhigh)
            @inferred pdf(TriangularDist(za, zb, zc), zxac)
            @inferred pdf(TriangularDist(za, zb, zc), zxbc)
            @inferred cdf(TriangularDist(za, zb, zc), zxa)
            @inferred cdf(TriangularDist(za, zb, zc), zxb)
            @inferred cdf(TriangularDist(za, zb, zc), zxlow)
            @inferred cdf(TriangularDist(za, zb, zc), zxhigh)
            @inferred cdf(TriangularDist(za, zb, zc), zxac)
            @inferred cdf(TriangularDist(za, zb, zc), zxbc)
            @inferred mgf(TriangularDist(za, zb, zc), zt1)
            @inferred mgf(TriangularDist(za, zb, zc), zt2)
            @inferred cf(TriangularDist(za, zb, zc), zt1)
            @inferred cf(TriangularDist(za, zb, zc), zt2)
        end
    end
    
    @testset "type_stable_symmetric_triangle" begin
        a = 2
        b = 6
        c = 4
        xa = a
        xb = b
        xlow = a - ψ
        xhigh = b + ψ
        xac = middle(a, c)
        xbc = middle(b, c)
        t1 = 1
        t2 = 2

        for j = 1:n
            typep = typeslist[j, 1]
            typex = typeslist[j, 2]
            za = convertstr(typep, a)
            zb = convertstr(typep, b)
            zc = convertstr(typep, c)
            zxa = convertstr(typex, xa)
            zxb = convertstr(typex, xb)
            zxlow = convertstr(typex, xlow)
            zxhigh = convertstr(typex, xhigh)
            zxac = convertstr(typex, xac)
            zxbc = convertstr(typex, xbc)
            zt1 = convertstr(typex, t1)
            zt2 = convertstr(typex, t2)

            @inferred pdf(TriangularDist(za, zb, zc), zxa)
            @inferred pdf(TriangularDist(za, zb, zc), zxb)
            @inferred pdf(TriangularDist(za, zb, zc), zxlow)
            @inferred pdf(TriangularDist(za, zb, zc), zxhigh)
            @inferred pdf(TriangularDist(za, zb, zc), zxac)
            @inferred pdf(TriangularDist(za, zb, zc), zxbc)
            @inferred cdf(TriangularDist(za, zb, zc), zxa)
            @inferred cdf(TriangularDist(za, zb, zc), zxb)
            @inferred cdf(TriangularDist(za, zb, zc), zxlow)
            @inferred cdf(TriangularDist(za, zb, zc), zxhigh)
            @inferred cdf(TriangularDist(za, zb, zc), zxac)
            @inferred cdf(TriangularDist(za, zb, zc), zxbc)
            @inferred mgf(TriangularDist(za, zb, zc), zt1)
            @inferred mgf(TriangularDist(za, zb, zc), zt2)
            @inferred cf(TriangularDist(za, zb, zc), zt1)
            @inferred cf(TriangularDist(za, zb, zc), zt2)
        end
    end
    
    @testset "type_stable_right_triangle" begin
        a = 2
        b = 4
        c = 4
        xa = a
        xb = b
        xlow = a - ψ
        xhigh = b + ψ
        xac = middle(a, c)
        xbc = middle(b, c)
        t1 = 1
        t2 = 2

        for j = 1:n
            typep = typeslist[j, 1]
            typex = typeslist[j, 2]
            za = convertstr(typep, a)
            zb = convertstr(typep, b)
            zc = convertstr(typep, c)
            zxa = convertstr(typex, xa)
            zxb = convertstr(typex, xb)
            zxlow = convertstr(typex, xlow)
            zxhigh = convertstr(typex, xhigh)
            zxac = convertstr(typex, xac)
            zxbc = convertstr(typex, xbc)
            zt1 = convertstr(typex, t1)
            zt2 = convertstr(typex, t2)

            @inferred pdf(TriangularDist(za, zb, zc), zxa)
            @inferred pdf(TriangularDist(za, zb, zc), zxb)
            @inferred pdf(TriangularDist(za, zb, zc), zxlow)
            @inferred pdf(TriangularDist(za, zb, zc), zxhigh)
            @inferred pdf(TriangularDist(za, zb, zc), zxac)
            @inferred pdf(TriangularDist(za, zb, zc), zxbc)
            @inferred cdf(TriangularDist(za, zb, zc), zxa)
            @inferred cdf(TriangularDist(za, zb, zc), zxb)
            @inferred cdf(TriangularDist(za, zb, zc), zxlow)
            @inferred cdf(TriangularDist(za, zb, zc), zxhigh)
            @inferred cdf(TriangularDist(za, zb, zc), zxac)
            @inferred cdf(TriangularDist(za, zb, zc), zxbc)
            @inferred mgf(TriangularDist(za, zb, zc), zt1)
            @inferred mgf(TriangularDist(za, zb, zc), zt2)
            @inferred cf(TriangularDist(za, zb, zc), zt1)
            @inferred cf(TriangularDist(za, zb, zc), zt2)
        end
    end
    
    @testset "type_stable_left_triangle" begin
        a = 2
        b = 4
        c = 2
        xa = a
        xb = b
        xlow = a - ψ
        xhigh = b + ψ
        xac = middle(a, c)
        xbc = middle(b, c)
        t1 = 1
        t2 = 2

        for j = 1:n
            typep = typeslist[j, 1]
            typex = typeslist[j, 2]
            za = convertstr(typep, a)
            zb = convertstr(typep, b)
            zc = convertstr(typep, c)
            zxa = convertstr(typex, xa)
            zxb = convertstr(typex, xb)
            zxlow = convertstr(typex, xlow)
            zxhigh = convertstr(typex, xhigh)
            zxac = convertstr(typex, xac)
            zxbc = convertstr(typex, xbc)
            zt1 = convertstr(typex, t1)
            zt2 = convertstr(typex, t2)

            @inferred pdf(TriangularDist(za, zb, zc), zxa)
            @inferred pdf(TriangularDist(za, zb, zc), zxb)
            @inferred pdf(TriangularDist(za, zb, zc), zxlow)
            @inferred pdf(TriangularDist(za, zb, zc), zxhigh)
            @inferred pdf(TriangularDist(za, zb, zc), zxac)
            @inferred pdf(TriangularDist(za, zb, zc), zxbc)
            @inferred cdf(TriangularDist(za, zb, zc), zxa)
            @inferred cdf(TriangularDist(za, zb, zc), zxb)
            @inferred cdf(TriangularDist(za, zb, zc), zxlow)
            @inferred cdf(TriangularDist(za, zb, zc), zxhigh)
            @inferred cdf(TriangularDist(za, zb, zc), zxac)
            @inferred cdf(TriangularDist(za, zb, zc), zxbc)
            @inferred mgf(TriangularDist(za, zb, zc), zt1)
            @inferred mgf(TriangularDist(za, zb, zc), zt2)
            @inferred cf(TriangularDist(za, zb, zc), zt1)
            @inferred cf(TriangularDist(za, zb, zc), zt2)
        end
    end
    
end
