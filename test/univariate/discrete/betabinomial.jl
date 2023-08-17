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
end
