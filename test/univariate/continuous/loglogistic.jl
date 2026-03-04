using Distributions
using Test

import Optim

@testset "LogLogistic" begin
    @testset "Constructors" begin
        for T1 in (Int, Float32, Float64), T2 in (Int, Float32, Float64)
           d = @inferred(LogLogistic(T1(1), T2(2)))
           @test d isa LogLogistic{promote_type(T1, T2)}
           @test partype(d) === promote_type(T1, T2)
           @test d.α == 1
           @test d.β == 2

           @test_throws ArgumentError LogLogistic(T1(-1), T2(2))
           @test_throws ArgumentError LogLogistic(T1(-1), T2(2); check_args=true)
           d = @inferred(LogLogistic(T1(-1), T2(2); check_args=false))
           @test d isa LogLogistic{promote_type(T1, T2)}
           @test partype(d) === promote_type(T1, T2)
           @test d.α == -1
           @test d.β == 2

           @test_throws ArgumentError LogLogistic(T1(1), T2(-2))
           @test_throws ArgumentError LogLogistic(T1(1), T2(-2); check_args=true)
           d = @inferred(LogLogistic(T1(1), T2(-2); check_args=false))
           @test d isa LogLogistic{promote_type(T1, T2)}
           @test partype(d) === promote_type(T1, T2)
           @test d.α == 1
           @test d.β == -2
        end
    end

    @testset "Conversion" begin
        d = LogLogistic(1.0, 2.0)
        @test convert(LogLogistic{Float64}, d) === d
        @test convert(LogLogistic{Float32}, d) isa LogLogistic{Float32}
        @test convert(LogLogistic{Float32}, d) == d
    end

    @testset "median" begin
        for α in (0.5, 1, 2, 3), β in (0.5, 1, 2, 3)
            d = LogLogistic(α, β)
            @test median(d) ≈ quantile(d, 1//2)
        end
    end

    @testset "mode" begin
        for α in (0.5, 1, 2, 3), β in (0.5, 1, 2, 3)
            d = LogLogistic(α, β)
            opt = Optim.maximize(Base.Fix1(logpdf, d), 0.0, 10.0)
            @test mode(d) ≈ Optim.maximizer(opt) rtol = 1e-8 atol = 1e-12
        end
    end

    @testset "mean" begin
        for α in (0.5, 1, 2, 3), β in (0.5, 1, 2, 3)
            d = LogLogistic(α, β)
            if β > 1
                @test mean(d) ≈ Distributions.expectation(identity, d)
            else
                @test_throws ArgumentError("the mean of a log-logistic distribution is defined only when its shape β > 1") mean(d)
            end
        end
    end

    @testset "variance" begin
        for α in (0.5, 1, 2, 3), β in (0.5, 1, 2, 3)
            d = LogLogistic(α, β)
            if β > 2
                m = mean(d)
                @test var(d) ≈ Distributions.expectation(x -> (x - m)^2, d)
            else
                @test_throws ArgumentError("the variance of a log-logistic distribution is defined only when its shape β > 2") var(d)
            end
        end
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

    @testset "entropy" begin
        for α in (0.5, 1, 2, 3), β in (0.5, 1, 2, 3)
            d = LogLogistic(α, β)
            @test entropy(d) ≈ -Distributions.expectation(Base.Fix1(logpdf, d), d) rtol = 1e-6
        end
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
