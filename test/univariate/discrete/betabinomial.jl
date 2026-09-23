using Distributions
using Test

@testset "betabinomial.jl" begin
    @testset "logpdf" begin
        d = BetaBinomial(50, 0.2, 0.6)

        for k in 1:50
            p  = @inferred(pdf(d, k))
            lp = @inferred(logpdf(d, k))
            @test lp ≈ log(p)
        end
    end

    @testset "support" begin
        d = BetaBinomial(50, 0.2, 0.6)

        for k in 1:50
            @test insupport(d, k)
        end
        @test !insupport(d, 51)
    end

    @testset "checks" begin
        for T in (Int, Float64), S in (Int, Float64)
            ST = float(promote_type(S, T))
            for n in (-1, 0, 3), α in (S(-1), S(0), S(1)), β in (T(-1), T(0), T(1))
                if n >= 0 && α > 0 && β > 0
                    @test @inferred(BetaBinomial(n, α, β)) isa BetaBinomial{ST}
                    @test @inferred(BetaBinomial(n, α, β; check_args=true)) isa BetaBinomial{ST}
                else
                    @test_throws DomainError BetaBinomial(n, α, β)
                    @test_throws DomainError BetaBinomial(n, α, β; check_args=true)
                end

                @test @inferred(BetaBinomial(n, α, β; check_args=false)) isa BetaBinomial{ST}
            end
        end
    end

    @testset "quantile" begin
        # https://github.com/JuliaStats/Distributions.jl/issues/2090
        @test @inferred(quantile(BetaBinomial(1000, 0.1, 20), 1)) == 1000
        @test @inferred(quantile(BetaBinomial(100, 0.1, 10), 1)) == 100

        d = BetaBinomial(10, 2, 2.5)
        @test @inferred(quantile(d, 1//2)) == quantile(d, 0.5) == median(d)
        @test @inferred(quantile(d, 0.5f0)) isa Int
        for p in (-0.1, 1.1, NaN)
            err = DomainError(p, "`p` must satisfy `0 <= p <= 1`")
            @test_throws err quantile(d, p)
            @test_throws err cquantile(d, p)
        end

        @testset "$d" for d in (BetaBinomial(0, 1, 1), BetaBinomial(1, 0.5, 0.5),
                                BetaBinomial(10, 2, 2.5), BetaBinomial(50, 0.2, 0.6),
                                BetaBinomial(100, 0.1, 10), BetaBinomial(1000, 0.1, 20))
            test_quantile_invariants(d)
        end
    end

    @testset "mode" begin
        @testset "$d" for d in (BetaBinomial(0, 1, 1), BetaBinomial(1, 0.5, 0.5),
                                BetaBinomial(10, 2, 2.5), BetaBinomial(10, 60, 40),
                                BetaBinomial(10, 0.5, 0.5), BetaBinomial(11, 3, 3),
                                BetaBinomial(20, 3, 1), BetaBinomial(50, 0.2, 0.6),
                                BetaBinomial(1000, 0.1, 20))
            ps = map(Base.Fix1(pdf, d), support(d))
            @test @inferred(modes(d)) == support(d)[ps .== maximum(ps)]
            @test @inferred(mode(d)) == first(modes(d))
        end

        # `BetaBinomial(n, 1, 1)` is uniform, but `pdf` breaks the ties by rounding errors
        @test @inferred(modes(BetaBinomial(20, 1, 1))) == 0:20
        @test @inferred(mode(BetaBinomial(20, 1, 1))) == 0
    end
end
