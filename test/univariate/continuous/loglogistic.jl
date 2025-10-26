using Distributions
using Test

@testset "LogLogistic" begin
    @testset "Conversion" begin
        d = LogLogistic(1.0, 2.0)
        @test convert(LogLogistic{Float64}, d) === d
        @test convert(LogLogistic{Float32}, d) isa LogLogistic{Float32}
        @test convert(LogLogistic{Float32}, d) == d
    end

    # Values computed with WolframAlpha
    @testset "Special values" begin
        # pdf
        @test iszero(pdf(LogLogistic(1, 1), -1))
        @test pdf(LogLogistic(1, 1), 1) ≈ 0.25
        @test pdf(LogLogistic(2, 2), 1) ≈ 0.32
        @test pdf(LogLogistic(2, 2), 4) ≈ 0.08

        # log pdf
        @test logpdf(LogLogistic(1, 1), -1) == -Inf
        @test logpdf(LogLogistic(1, 1), 1) ≈ -log(4)
        @test logpdf(LogLogistic(2, 2), 1) ≈ log(0.32)
        @test logpdf(LogLogistic(2, 2), 4) ≈ log(0.08)

        # cdf
        @test iszero(cdf(LogLogistic(1, 1), -1))
        @test cdf(LogLogistic(1, 1), Inf) == 1
        @test cdf(LogLogistic(1, 1), 1) ≈ 0.5
        @test cdf(LogLogistic(2, 2), 1) ≈ 0.2
        @test cdf(LogLogistic(2, 2), 4) ≈ 0.8

        # log cdf
        @test logcdf(LogLogistic(1, 1), -1) == -Inf
        @test iszero(logcdf(LogLogistic(1, 1), Inf))
        @test logcdf(LogLogistic(1, 1), 1) ≈ -log(2)
        @test logcdf(LogLogistic(2, 2), 1) ≈ log(0.2)
        @test logcdf(LogLogistic(2, 2), 4) ≈ log(0.8)

        # ccdf
        @test ccdf(LogLogistic(1, 1), -1) == 1
        @test iszero(ccdf(LogLogistic(1, 1), Inf))
        @test ccdf(LogLogistic(1, 1), 1) ≈ 0.5
        @test ccdf(LogLogistic(2, 2), 1) ≈ 0.8
        @test ccdf(LogLogistic(2, 2), 4) ≈ 0.2

        # log ccdf
        @test iszero(logccdf(LogLogistic(1, 1), -1))
        @test logccdf(LogLogistic(1, 1), Inf) == -Inf
        @test logccdf(LogLogistic(1, 1), 1) ≈ -log(2)
        @test logccdf(LogLogistic(2, 2), 1) ≈ log(0.8)
        @test logccdf(LogLogistic(2, 2), 4) ≈ log(0.2)
    end

    @testset "Default tests" begin
        for d in [
            LogLogistic(1, 1),
            LogLogistic(2, 1),
            LogLogistic(2, 2),
        ]
            test_distr(d, 10^6)
        end
    end
end
