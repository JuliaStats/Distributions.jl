using Test, Distributions, SpecialFunctions

@testset "pdf L2 norm" begin
    # Test error on a non implemented norm.
    @test_throws MethodError pdfsquaredL2norm(Gumbel())

    @testset "Beta" begin
        @test pdfsquaredL2norm(Beta(1, 1)) ≈ 1
        @test pdfsquaredL2norm(Beta(2, 2)) ≈ 6 / 5
        @test pdfsquaredL2norm(Beta(0.25, 1)) ≈ Inf
        @test pdfsquaredL2norm(Beta(1, 0.25)) ≈ Inf
    end

    @testset "Cauchy" begin
        @test pdfsquaredL2norm(Cauchy(0, 1)) ≈ 1 / (2 * π)
        @test pdfsquaredL2norm(Cauchy(0, 2)) ≈ 1 / (4 * π)
        # The norm doesn't depend on the mean
        @test pdfsquaredL2norm(Cauchy(100, 1)) == pdfsquaredL2norm(Cauchy(-100, 1)) == pdfsquaredL2norm(Cauchy(0, 1))
    end

    @testset "Chi" begin
        @test pdfsquaredL2norm(Chi(2)) ≈ gamma(3 / 2) / 2
        @test pdfsquaredL2norm(Chi(0.25)) ≈ Inf
    end

    @testset "Chisq" begin
        @test pdfsquaredL2norm(Chisq(2)) ≈ 1 / 4
        @test pdfsquaredL2norm(Chisq(1)) ≈ Inf
    end

    @testset "Exponential" begin
        @test pdfsquaredL2norm(Exponential(1)) ≈ 1 / 2
        @test pdfsquaredL2norm(Exponential(2)) ≈ 1 / 4
    end

    @testset "Gamma" begin
        @test pdfsquaredL2norm(Gamma(1, 1)) ≈ 1 / 2
        @test pdfsquaredL2norm(Gamma(1, 2)) ≈ 1 / 4
        @test pdfsquaredL2norm(Gamma(2, 2)) ≈ 1 / 8
        @test pdfsquaredL2norm(Gamma(1, 0.25)) ≈ 2
        @test pdfsquaredL2norm(Gamma(0.5, 1)) ≈ Inf
    end

    @testset "Logistic" begin
        @test pdfsquaredL2norm(Logistic(0, 1)) ≈ 1 / 6
        @test pdfsquaredL2norm(Logistic(0, 2)) ≈ 1 / 12
        # The norm doesn't depend on the mean
        @test pdfsquaredL2norm(Logistic(100, 1)) == pdfsquaredL2norm(Logistic(-100, 1)) == pdfsquaredL2norm(Logistic(0, 1))
    end

    @testset "Normal" begin
        @test pdfsquaredL2norm(Normal(0, 1)) ≈ 1 / (2 * sqrt(π))
        @test pdfsquaredL2norm(Normal(0, 2)) ≈ 1 / (4 * sqrt(π))
        @test pdfsquaredL2norm(Normal(1, 0)) ≈ Inf
        # The norm doesn't depend on the mean
        @test pdfsquaredL2norm(Normal(100, 1)) == pdfsquaredL2norm(Normal(-100, 1)) == pdfsquaredL2norm(Normal(0, 1))
    end

    @testset "Uniform" begin
        @test pdfsquaredL2norm(Uniform(-1, 1)) ≈ 1 / 2
        @test pdfsquaredL2norm(Uniform(1, 2)) ≈ 1
    end
end
