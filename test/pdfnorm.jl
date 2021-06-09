using Test, Distributions, SpecialFunctions
using QuadGK

# `numeric_norm` is a helper function to compute numerically the squared L2
# norms of the distributions.  These methods aren't very robust because can't
# deal with divergent norms, or discrete distributions with infinite support.
numeric_norm(d::ContinuousUnivariateDistribution) =
    quadgk(x -> pdf(d, x) ^ 2, support(d).lb, support(d).ub)[1]

function numeric_norm(d::DiscreteUnivariateDistribution)
    # When the distribution has infinite support, sum up to an arbitrary large
    # value.
    upper = isfinite(maximum(d)) ? round(Int, maximum(d)) : 100
    return sum(pdf(d, k) ^ 2 for k in round(Int, minimum(d)):upper)
end

@testset "pdf L2 norm" begin
    # Test error on a non implemented norm.
    @test_throws MethodError pdfsquaredL2norm(Gumbel())

    @testset "Bernoulli" begin
        for d in (Bernoulli(0.5), Bernoulli(0), Bernoulli(0.25))
            @test pdfsquaredL2norm(d) ≈ numeric_norm(d)
        end
        # The norm is the same for complementary probabilities
        @test pdfsquaredL2norm(Bernoulli(0)) == pdfsquaredL2norm(Bernoulli(1))
        @test pdfsquaredL2norm(Bernoulli(0.25)) == pdfsquaredL2norm(Bernoulli(0.75))
    end

    @testset "Beta" begin
        for d in (Beta(1, 1), Beta(2, 2))
            @test pdfsquaredL2norm(d) ≈ numeric_norm(d)
        end
        @test pdfsquaredL2norm(Beta(0.25, 1)) ≈ Inf
        @test pdfsquaredL2norm(Beta(1, 0.25)) ≈ Inf
    end

    @testset "Categorical" begin
        for n in (1, 2, 5, 10)
            d = Categorical(collect(1 / n for _ in 1:n))
            @test pdfsquaredL2norm(d) ≈ numeric_norm(d)
        end
        for d in (Categorical([0.25, 0.75]), Categorical([1 / 6, 1 / 3, 1 / 2]))
            @test pdfsquaredL2norm(d) ≈ numeric_norm(d)
        end
    end

    @testset "Cauchy" begin
        for d in (Cauchy(0, 1), Cauchy(0, 2))
            @test pdfsquaredL2norm(d) ≈ numeric_norm(d)
        end
        # The norm doesn't depend on the mean
        @test pdfsquaredL2norm(Cauchy(100, 1)) == pdfsquaredL2norm(Cauchy(-100, 1)) == pdfsquaredL2norm(Cauchy(0, 1))
    end

    @testset "Chi" begin
        @test pdfsquaredL2norm(Chi(2)) ≈ numeric_norm(Chi(2))
        @test pdfsquaredL2norm(Chi(0.25)) ≈ Inf
    end

    @testset "Chisq" begin
        @test pdfsquaredL2norm(Chisq(2)) ≈ numeric_norm(Chisq(2))
        @test pdfsquaredL2norm(Chisq(1)) ≈ Inf
    end

    @testset "DiscreteUniform" begin
        for d in (DiscreteUniform(-1, 1), DiscreteUniform(1, 2))
            @test pdfsquaredL2norm(d) ≈ numeric_norm(d)
        end
    end

    @testset "Exponential" begin
        for d in (Exponential(1), Exponential(2))
            @test pdfsquaredL2norm(d) ≈ numeric_norm(d)
        end
    end

    @testset "Gamma" begin
        for d in (Gamma(1, 1), Gamma(1, 2), Gamma(2, 2), Gamma(1, 0.25))
            @test pdfsquaredL2norm(d) ≈ numeric_norm(d)
        end
        @test pdfsquaredL2norm(Gamma(0.5, 1)) ≈ Inf
    end

    @testset "Geometric" begin
        for d in (Geometric(0.20), Geometric(0.25), Geometric(0.50), Geometric(0.75), Geometric(0.80))
            @test pdfsquaredL2norm(d) ≈ numeric_norm(d)
        end
    end

    @testset "Logistic" begin
        for d in (Logistic(0, 1), Logistic(0, 2))
            @test pdfsquaredL2norm(d) ≈ numeric_norm(d)
        end
        # The norm doesn't depend on the mean
        @test pdfsquaredL2norm(Logistic(100, 1)) == pdfsquaredL2norm(Logistic(-100, 1)) == pdfsquaredL2norm(Logistic(0, 1))
    end

    @testset "Normal" begin
        for d in (Normal(0, 1), Normal(0, 2))
            @test pdfsquaredL2norm(d) ≈ numeric_norm(d)
        end
        @test pdfsquaredL2norm(Normal(1, 0)) ≈ Inf
        # The norm doesn't depend on the mean
        @test pdfsquaredL2norm(Normal(100, 1)) == pdfsquaredL2norm(Normal(-100, 1)) == pdfsquaredL2norm(Normal(0, 1))
    end

    @testset "Poisson" begin
        for d in (Poisson(0), Poisson(1), Poisson(pi))
            @test pdfsquaredL2norm(d) ≈ numeric_norm(d)
        end
    end

    @testset "Uniform" begin
        for d in (Uniform(-1, 1), Uniform(1, 2))
            @test pdfsquaredL2norm(d) ≈ numeric_norm(d)
        end
    end
end
