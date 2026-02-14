using Distributions, Test

@testset "Levy" begin
    @testset "values outside of the support" begin
        for (d, x) in ((Levy(2), 1.8), (Levy(2, 8), 1.8), (Levy(3, 3), 2.8))
            @test iszero(@inferred(pdf(d, x)))
            @test @inferred(logpdf(d, x)) == -Inf
            @test iszero(@inferred(cdf(d, x)))
            @test @inferred(logcdf(d, x)) == -Inf
            @test isone(@inferred(ccdf(d, x)))
            @test iszero(@inferred(logccdf(d, x)))
        end
    end
end
