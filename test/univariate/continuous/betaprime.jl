using Distributions, Test

@testset "BetaPrime" begin
    @testset "values outside of the support" begin
        d = BetaPrime()
        x = -2.25
        @test iszero(@inferred(pdf(d, x)))
        @test @inferred(logpdf(d, x)) == -Inf
        @test iszero(@inferred(cdf(d, x)))
        @test @inferred(logcdf(d, x)) == -Inf
        @test isone(@inferred(ccdf(d, x)))
        @test iszero(@inferred(logccdf(d, x)))
    end
end
