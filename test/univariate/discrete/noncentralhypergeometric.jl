using Distributions
using Test

# `pdf(d, k) ∝ binomial(ns, k) * binomial(nf, n - k) * ω^k`, evaluated exactly
function fisher_refprobs(d::FisherNoncentralHypergeometric)
    ns, nf, n, ω = params(d)
    ws = [binomial(big(ns), big(k)) * binomial(big(nf), big(n - k)) * big(ω)^k
          for k in support(d)]
    return ws ./ sum(ws)
end

@testset "noncentralhypergeometric" begin
    @testset "$(nameof(D))" for D in (FisherNoncentralHypergeometric,
                                      WalleniusNoncentralHypergeometric)
        d = D(8, 6, 10, 2)
        xs = support(d)

        @testset "constructor" begin
            @test D(8, 6, 10, 1) isa D{Float64}
            @test D(8, 6, 10, 0.5f0) isa D{Float32}
            @test params(d) == (8, 6, 10, 2.0)
            @test partype(d) === Float64
            for args in ((-1, 6, 10, 1), (8, -1, 10, 1), (8, 6, 0, 1), (8, 6, 14, 1),
                         (8, 6, 10, 0))
                @test_throws DomainError D(args...)
                @test D(args...; check_args=false) isa D{Float64}
            end
        end

        @testset "support" begin
            @test extrema(d) == (4, 8)
            @test xs == 4:8
            @test insupport(d, 4)
            @test !insupport(d, 3)
            @test !insupport(d, 4.5)
            # the support starts at zero unless more than `nf` samples are drawn
            @test support(D(50, 500, 30, 0.3)) == 0:30
        end

        @testset "pdf" begin
            @test sum(Base.Fix1(pdf, d), xs) ≈ 1
            @test all(pdf(d, k) == pdf(d, float(k)) == pdf(d, k // 1) for k in xs)
            @test all(logpdf(d, k) ≈ log(pdf(d, k)) for k in xs)

            # `pdf(::FisherNoncentralHypergeometric, ::AbstractFloat)` used to recurse
            for x in (4.5, 3, 9, 3.0, 9.0, -Inf, Inf)
                @test iszero(pdf(d, x))
                @test logpdf(d, x) == -Inf
            end
            @test isnan(pdf(d, NaN))
            @test isnan(logpdf(d, NaN))
        end

        @testset "parameter type" begin
            d32 = D(8, 6, 10, 2.0f0)
            # the Wallenius pmf loses the parameter type in the numerical integration
            broken = D === WalleniusNoncentralHypergeometric
            for f in (pdf, logpdf, cdf, ccdf, logcdf, logccdf)
                @test @inferred(f(d32, 5)) isa Float32 broken=broken
                @test @inferred(f(d32, 5.0)) isa Float32 broken=broken
            end
            @test @inferred(quantile(d32, 0.5)) isa Int
            @test @inferred(cquantile(d32, 0.5)) isa Int
        end

        @testset "cdf" begin
            cs = map(Base.Fix1(cdf, d), xs)
            gs = map(Base.Fix1(ccdf, d), xs)
            @test issorted(cs)
            @test issorted(gs; rev=true)
            @test all(cs .+ gs .≈ 1)
            @test cs ≈ cumsum(map(Base.Fix1(pdf, d), xs))
            @test all(logcdf(d, k) ≈ log(cdf(d, k)) for k in xs)
            @test all(logccdf(d, k) ≈ log(ccdf(d, k)) for k in xs)

            @test iszero(cdf(d, -Inf))
            @test iszero(cdf(d, 3))
            @test isone(cdf(d, 8))
            @test isone(cdf(d, Inf))
            @test isone(ccdf(d, -Inf))
            @test iszero(ccdf(d, 8))
            @test iszero(ccdf(d, Inf))
            for f in (cdf, ccdf, logcdf, logccdf)
                @test isnan(f(d, NaN))
                @test f(d, 5.5) == f(d, 5)
            end
        end

        # `ω = 1` reduces to the hypergeometric distribution, up to the accuracy of the
        # numerical integration in the Wallenius pmf
        @testset "hypergeometric limit" begin
            d1 = D(8, 6, 10, 1)
            h = Hypergeometric(8, 6, 10)
            rtol = D === FisherNoncentralHypergeometric ? 1e-13 : 1e-7
            @test mean(d1) ≈ mean(h) rtol=rtol
            @test var(d1) ≈ var(h) rtol=rtol
            @test mode(d1) == mode(h)
            @test all(isapprox(pdf(d1, k), pdf(h, k); rtol=rtol) for k in support(h))
            @test all(isapprox(cdf(d1, k), cdf(h, k); rtol=rtol) for k in support(h))
        end

        @testset "quantile" begin
            # https://github.com/JuliaStats/Distributions.jl/issues/2090
            # `quantile(d, 0)` and `quantile(d, 1)` used to throw a `BoundsError`
            @testset "$dq" for dq in (D(8, 6, 10, 1), d, D(50, 500, 30, 0.3))
                @test @inferred(quantile(dq, 0)) == minimum(dq)
                @test @inferred(quantile(dq, 1)) == maximum(dq)
                @test @inferred(cquantile(dq, 0)) == maximum(dq)
                @test @inferred(cquantile(dq, 1)) == minimum(dq)
                @test @inferred(quantile(dq, 1 // 2)) == quantile(dq, 0.5) == median(dq)
                @test @inferred(quantile(dq, 0.5f0)) isa Int
                test_quantile_invariants(dq)
            end

            for p in (-0.1, 1.1, NaN)
                err = DomainError(p, "`p` must satisfy `0 <= p <= 1`")
                @test_throws err quantile(d, p)
                @test_throws err cquantile(d, p)
            end
        end
    end

    @testset "Fisher: exact reference probabilities" begin
        @testset "$d" for d in (FisherNoncentralHypergeometric(8, 6, 10, 2),
                                FisherNoncentralHypergeometric(50, 500, 30, 0.3),
                                FisherNoncentralHypergeometric(80, 60, 100, 10))
            ps = fisher_refprobs(d)
            xs = support(d)
            @test all(isapprox(pdf(d, k), ps[i]; rtol=1e-13)
                      for (i, k) in enumerate(xs))
            @test all(isapprox(cdf(d, k), sum(ps[1:i]); rtol=1e-13)
                      for (i, k) in enumerate(xs))
            @test all(isapprox(ccdf(d, k), sum(ps[(i + 1):end]); rtol=1e-13)
                      for (i, k) in enumerate(xs) if k < last(xs))
            @test iszero(ccdf(d, last(xs)))

            μ = sum(k * ps[i] for (i, k) in enumerate(xs))
            @test mean(d) ≈ μ rtol=1e-12
            @test var(d) ≈ sum((k - μ)^2 * ps[i] for (i, k) in enumerate(xs)) rtol=1e-12
        end

        # `1 - cdf(d, k)` cancels to zero in the upper tail
        d = FisherNoncentralHypergeometric(50, 500, 30, 0.3)
        @test iszero(1 - cdf(d, 29))
        @test isfinite(logccdf(d, 29))

        # terms in the lower tail are negligible relative to the normalizing constant, but
        # not relative to `cdf(d, k)`
        d = FisherNoncentralHypergeometric(80, 60, 100, 10)
        @test cdf(d, 51) > pdf(d, 51)
    end
end
