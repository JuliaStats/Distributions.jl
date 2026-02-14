using Distributions, Test

@testset "StudentizedRange" begin
    @testset "values at 0" begin
        d = StudentizedRange(2, 2)
        @test @inferred(pdf(d, 0)) ≈ 0.5
        @test @inferred(logpdf(d, 0)) ≈ -log(2)
        @test iszero(@inferred(cdf(d, 0)))
        @test @inferred(logcdf(d, 0)) == -Inf
        @test isone(@inferred(ccdf(d, 0)))
        @test iszero(@inferred(logccdf(d, 0)))

        for d in (StudentizedRange(10, 5), StudentizedRange(5, 10))
            @test iszero(@inferred(pdf(d, 0)))
            @test @inferred(logpdf(d, 0)) == -Inf
            @test iszero(@inferred(cdf(d, 0)))
            @test @inferred(logcdf(d, 0)) == -Inf
            @test isone(@inferred(ccdf(d, 0)))
            @test iszero(@inferred(logccdf(d, 0)))
        end
    end
end
