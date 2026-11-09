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
