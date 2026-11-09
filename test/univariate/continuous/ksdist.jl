using Distributions
using Test

# Exact P(D_n ≤ x) from the Durbin matrix of Marsaglia, Tsang and Wang (2003),
# evaluated in rational arithmetic at the exact value of `x`
function ksdist_cdf_exact(n::Int, x::Float64)
    t = Rational{BigInt}(x) * n
    k = Int(ceil(t))
    h = k - t
    m = 2k - 1
    H = [i - j + 1 >= 0 ? one(t) : zero(t) for i in 1:m, j in 1:m]
    for i in 1:m
        H[i, 1] -= h^i
        H[m, i] -= h^(m - i + 1)
    end
    if 2h > 1
        H[m, 1] += (2h - 1)^m
    end
    for i in 1:m, j in 1:m
        H[i, j] /= factorial(big(max(i - j + 1, 0)))
    end
    return (H^n)[k, k] * factorial(big(n)) // big(n)^n
end

@testset "KSDist" begin
    @testset "issue #2111 and #254" begin
        @test cdf(KSDist(3), 0.4) ≈ 0.40533333333333339 rtol = 1e-14
        @test cdf(KSDist(5), 0.25) ≈ 0.1554 rtol = 1e-14
        @test cdf(KSDist(15), 0.084061515) ≈ 0.00043984029545340023 rtol = 1e-14
        @test cdf(KSDist(5), 2.256281e-01) ≈ 0.08852108277095537 rtol = 1e-14
    end

    # Polynomials for n = 10 from p. 2 of Marsaglia, Tsang and Wang (2003)
    @testset "Marsaglia, Tsang and Wang (2003)" begin
        d = KSDist(10)
        # 5/20 < x < 6/20 (h ≤ 1/2)
        p1(x) = 419328x^10 - 801024x^9 + 3771936//5 * x^8 - 11684736//25 * x^7 +
            24769584//125 * x^6 - 32213664//625 * x^5 + 3604041//625 * x^4 +
            5313231//12500 * x^3 - 7515459//50000 * x^2 + 25247817//2500000 * x -
            15369417//100000000
        # 4/20 < x < 5/20 (h > 1/2)
        p2(x) = -806400x^10 + 1102080x^9 - 594720x^8 + 177408//5 * x^7 +
            3421908//25 * x^6 - 9773694//125 * x^5 + 47717019//2500 * x^4 -
            13212297//6250 * x^3 + 1035279//12500 * x^2 + 848673//625000 * x -
            88389//781250
        @test cdf(d, 0.274) ≈ 0.6284796154565043 rtol = 1e-14
        for (p, lo, hi) in ((p1, 5 / 20, 6 / 20), (p2, 4 / 20, 5 / 20))
            for x in range(lo, hi; length = 12)[2:(end - 1)]
                @test cdf(d, x) ≈ p(Rational{BigInt}(x)) rtol = 1e-14
            end
        end
    end

    @testset "exact values (n = $n)" for n in (3, 4, 5, 6, 8, 10)
        d = KSDist(n)
        @test iszero(cdf(d, 1 / (2n)))
        @test isone(ccdf(d, 1 / (2n)))
        @test isone(cdf(d, 1.0))
        @test iszero(ccdf(d, 1.0))

        # both sides of frac(n * x) = 1/2, in all regimes of `cdf` and `ccdf`
        for j in 0:(n - 1), f in (0.1, 0.25, 0.75, 0.9)
            x = (j + f) / n
            n * x > 1 / 2 || continue
            p = ksdist_cdf_exact(n, x)
            @test cdf(d, x) ≈ p rtol = 1e-12
            @test ccdf(d, x) ≈ 1 - p rtol = 1e-10
        end
    end
end
