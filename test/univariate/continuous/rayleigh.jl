using Distributions, Test

@testset "Rayleigh" begin
    @testset "values outside of the support" begin
        x = -0.1
        for d in (Rayleigh(), Rayleigh(3.0), Rayleigh(8))
            @test iszero(@inferred(pdf(d, x)))
            @test @inferred(logpdf(d, x)) == -Inf
            @test iszero(@inferred(cdf(d, x)))
            @test @inferred(logcdf(d, x)) == -Inf
            @test isone(@inferred(ccdf(d, x)))
            @test iszero(@inferred(logccdf(d, x)))
        end
    end
end
