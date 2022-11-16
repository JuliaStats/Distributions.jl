using Distributions

using Test

@testset "weibull.jl" begin
    # issue #1638
    @testset "pdf/logpdf: special cases" begin
        for S in (Float32, Float64), T in (Int, Float64, Float32)
            for α in (1, 2, 2.5), θ in (1, 1.5, 3)
                d = Weibull(S(α), S(θ))
                ST = promote_type(S, T)

                @test @inferred(pdf(d, T(-3))) === ST(0)
                @test @inferred(logpdf(d, T(-3))) === ST(-Inf)
                @test @inferred(pdf(d, T(0))) === (α == 1 ? ST(S(α) / S(θ)) : ST(0))
                @test @inferred(logpdf(d, T(0))) === (α == 1 ? ST(log(S(α) / S(θ))) : ST(-Inf))

                if T <: AbstractFloat
                    @test @inferred(pdf(d, T(-Inf))) === ST(0)
                    @test @inferred(logpdf(d, T(-Inf))) === ST(-Inf)
                    @test @inferred(pdf(d, T(Inf))) === ST(0)
                    @test @inferred(logpdf(d, T(Inf))) === ST(-Inf)
                    @test isnan(@inferred(pdf(d, T(NaN))))
                    @test isnan(@inferred(logpdf(d, T(NaN))))
                end
            end
        end
    end
end
