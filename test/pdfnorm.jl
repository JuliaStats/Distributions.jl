using Test, Distributions, SpecialFunctions

@testset "pdf L2 norm" begin
    # Test error on a non implemented norm.
    @test_throws MethodError pdfsquaredL2norm(Gumbel())

    @testset "Bernoulli" begin
        @test pdfsquaredL2norm(Bernoulli(0.5)) ≈ 0.5
        # The norm is the same for complementary probabilities
        @test pdfsquaredL2norm(Bernoulli(0)) == pdfsquaredL2norm(Bernoulli(1)) ≈ 1
        @test pdfsquaredL2norm(Bernoulli(0.25)) == pdfsquaredL2norm(Bernoulli(0.75)) ≈ 0.625
    end

    @testset "Beta" begin
        @test pdfsquaredL2norm(Beta(1, 1)) ≈ 1
        @test pdfsquaredL2norm(Beta(2, 2)) ≈ 6 / 5
        @test pdfsquaredL2norm(Beta(0.25, 1)) ≈ Inf
        @test pdfsquaredL2norm(Beta(1, 0.25)) ≈ Inf
    end

    @testset "Categorical" begin
        for n in (1, 2, 5, 10)
            @test pdfsquaredL2norm(Categorical(collect(1 / n for _ in 1:n))) ≈ 1 / n
        end
        @test pdfsquaredL2norm(Categorical([0.25, 0.75])) ≈ 0.625
        @test pdfsquaredL2norm(Categorical([1 / 6, 1 / 3, 1 / 2])) ≈ 7 / 18
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

    @testset "DiscreteUniform" begin
        @test pdfsquaredL2norm(DiscreteUniform(-1, 1)) ≈ 1 / 3
        @test pdfsquaredL2norm(DiscreteUniform(1, 2)) ≈ 1 / 2
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

    @testset "Geometric" begin
        @test pdfsquaredL2norm(Geometric(0.20)) ≈ 1 / 9
        @test pdfsquaredL2norm(Geometric(0.25)) ≈ 1 / 7
        @test pdfsquaredL2norm(Geometric(0.50)) ≈ 1 / 3
        @test pdfsquaredL2norm(Geometric(0.75)) ≈ 3 / 5
        @test pdfsquaredL2norm(Geometric(0.80)) ≈ 2 / 3
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

    @testset "Poisson" begin
        @test pdfsquaredL2norm(Poisson(0)) ≈ 1
        @test pdfsquaredL2norm(Poisson(1)) ≈ besseli(0, 2) * exp(-2)
        @test pdfsquaredL2norm(Poisson(pi)) ≈ besseli(0, 2 * pi) * exp(-2)
    end

    @testset "Uniform" begin
        @test pdfsquaredL2norm(Uniform(-1, 1)) ≈ 1 / 2
        @test pdfsquaredL2norm(Uniform(1, 2)) ≈ 1
    end
end
