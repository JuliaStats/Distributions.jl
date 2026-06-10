using Distributions, Test

@testset "StudentizedRange" begin
    @testset "values at 0" begin
        for d in (StudentizedRange(2, 2), StudentizedRange(10, 5), StudentizedRange(5, 10))
            @test iszero(@inferred(cdf(d, 0)))
            @test @inferred(logcdf(d, 0)) == -Inf
            @test isone(@inferred(ccdf(d, 0)))
            @test iszero(@inferred(logccdf(d, 0)))
        end
    end
end
