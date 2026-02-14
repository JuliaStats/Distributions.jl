using Distributions, Test

@testset "Cosine" begin
    @testset "values outside of the support" begin
        for (d, xmin, xmax) in ((Cosine(), -1.1, 1.1), (Cosine(1.5), -1.0, 3.0), (Cosine(2.0, 2.8), -0.9, 5.0))
            for x in (xmin, xmax)
                @test iszero(@inferred(pdf(d, x)))
                @test @inferred(logpdf(d, x)) == -Inf
            end

            @test iszero(@inferred(cdf(d, xmin)))
            @test @inferred(logcdf(d, xmin)) == -Inf
            @test isone(@inferred(ccdf(d, xmin)))
            @test iszero(@inferred(logccdf(d, xmin)))

            @test isone(@inferred(cdf(d, xmax)))
            @test iszero(@inferred(logcdf(d, xmax)))
            @test iszero(@inferred(ccdf(d, xmax)))
            @test @inferred(logccdf(d, xmax)) == -Inf
        end
    end
end
