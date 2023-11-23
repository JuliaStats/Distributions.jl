# Struct to test AbstractMvNormal methods
struct CholeskyMvNormal{M,T} <: Distributions.AbstractMvNormal
    m::M
    L::T
end

# Constructor for diagonal covariance matrices used in the tests below
function CholeskyMvNormal(m::Vector, Σ::Diagonal)
    L = Diagonal(map(sqrt, Σ.diag))
    return CholeskyMvNormal{typeof(m),typeof(L)}(m, L)
end

Distributions.length(p::CholeskyMvNormal) = length(p.m)
Distributions.mean(p::CholeskyMvNormal) = p.m
Distributions.cov(p::CholeskyMvNormal) = p.L * p.L'
Distributions.logdetcov(p::CholeskyMvNormal) = 2 * logdet(p.L)
function Distributions.sqmahal(p::CholeskyMvNormal, x::AbstractVector)
    return sum(abs2, p.L \ (mean(p) - x))
end
function Distributions._rand!(rng::AbstractRNG, p::CholeskyMvNormal, x::Vector)
    return x .= p.m .+ p.L * randn!(rng, x)
end

@testset "Expectations" begin
    # univariate distributions
    for d in (Normal(), Poisson(2.0), Binomial(10, 0.4))
        m = Distributions.expectation(identity, d)
        @test m ≈ mean(d) atol=1e-3
        @test Distributions.expectation(x -> (x - mean(d))^2, d) ≈ var(d) atol=1e-3

        @test @test_deprecated(Distributions.expectation(d, identity, 1e-10)) == m
        @test @test_deprecated(Distributions.expectation(d, identity)) == m
    end

    # multivariate distribution
    d = MvNormal([1.5, -0.5], I)
    @test Distributions.expectation(identity, d; nsamples=10_000) ≈ mean(d) atol=5e-2
    @test @test_deprecated(Distributions.expectation(d, identity; nsamples=10_000)) ≈ mean(d) atol=5e-2
end

@testset "KL divergences" begin
    function test_kl(p, q)
        @test kldivergence(p, q) >= 0
        @test kldivergence(p, p) ≈ 0 atol=1e-1
        @test kldivergence(q, q) ≈ 0 atol=1e-1
        if p isa UnivariateDistribution
            @test kldivergence(p, q) ≈ invoke(kldivergence, Tuple{UnivariateDistribution,UnivariateDistribution}, p, q) atol=1e-1
        elseif p isa MultivariateDistribution
            @test kldivergence(p, q) ≈ invoke(kldivergence, Tuple{MultivariateDistribution,MultivariateDistribution}, p, q; nsamples=10000) atol=1e-1
        end
    end

    @testset "univariate" begin
        @testset "Beta" begin
            p = Beta(2, 10)
            q = Beta(3, 5)
            test_kl(p, q)
        end
        @testset "Binomial" begin
            p = Binomial(3, 0.3)
            q = Binomial(3, 0.5)
            test_kl(p, q)
            @test iszero(kldivergence(Binomial(0, 0), Binomial(0, 1)))
            @test iszero(kldivergence(Binomial(0, 0.5), Binomial(0, 0.3)))
            @test isinf(kldivergence(Binomial(4, 0.3), Binomial(2, 0.3)))
            @test isinf(kldivergence(Binomial(3, 0), Binomial(3, 1)))
            @test isinf(kldivergence(Binomial(3, 0), Binomial(5, 1)))
            @test kldivergence(p, q) ≈ 3 * kldivergence(Bernoulli(0.3), Bernoulli(0.5))
        end
        @testset "Categorical" begin
            @test kldivergence(Categorical([0.0, 0.1, 0.9]), Categorical([0.1, 0.1, 0.8])) ≥ 0
            @test kldivergence(Categorical([0.0, 0.1, 0.9]), Categorical([0.1, 0.1, 0.8])) ≈
                kldivergence([0.0, 0.1, 0.9], [0.1, 0.1, 0.8])
        end
        @testset "Chi" begin
            p = Chi(4.0)
            q = Chi(3.0)
            test_kl(p, q)
            @test kldivergence(p, q) ≈ kldivergence(Gamma(2., 0.5), Gamma(1.5, 0.5))
        end
        @testset "Chisq" begin
            p = Chisq(4.0)
            q = Chisq(3.0)
            test_kl(p, q)
            @test kldivergence(p, q) ≈ kldivergence(Chi(4.0), Chi(3.0))
            @test kldivergence(p, q) ≈ kldivergence(Gamma(2., 0.5), Gamma(1.5, 0.5))
        end
        @testset "Exponential" begin
            p = Exponential(2.0)
            q = Exponential(3.0)
            test_kl(p, q)
        end
        @testset "Gamma" begin
            p = Gamma(2.0, 1.0)
            q = Gamma(3.0, 2.0)
            test_kl(p, q)
        end
        @testset "Geometric" begin
            p = Geometric(0.3)
            q = Geometric(0.4)
            test_kl(p, q)
            
            x1 = nextfloat(0.0)
            x2 = prevfloat(1.0)
            p1 = Geometric(x1)
            p2 = Geometric(x2)
            @test iszero(kldivergence(p2, p2))
            @test iszero(kldivergence(p1, p1))
            @test isinf(kldivergence(p1, p2))
            @test kldivergence(p2, p1) ≈ -log(x1)
            @test isinf(kldivergence(p1, Geometric(0.5)))
            @test kldivergence(p2, Geometric(0.5)) ≈ -log(0.5)
            @test kldivergence(Geometric(0.5), p2) ≈ 2*log(0.5) - log(1-x2)
            @test kldivergence(Geometric(0.5), p1) ≈ 2*log(0.5) - log(x1)
        end
        @testset "InverseGamma" begin
            p = InverseGamma(2.0, 1.0)
            q = InverseGamma(3.0, 2.0)
            test_kl(p, q)
        end
        @testset "Laplace" begin
            p = Laplace(2.0)
            q = Laplace(3.0)
            test_kl(p, q)
        end
        @testset "LogNormal" begin
            p = LogNormal(0, 1)
            q = LogNormal(0.5, 0.5)
            test_kl(p, q)
            @test kldivergence(p, q) ≈ kldivergence(Normal(0, 1), Normal(0.5, 0.5))
        end
        @testset "LogitNormal" begin
            p = LogitNormal(0, 1)
            q = LogitNormal(0.5, 0.5)
            test_kl(p, q)
            @test kldivergence(p, q) ≈ kldivergence(Normal(0, 1), Normal(0.5, 0.5))
        end
        @testset "NegativeBinomial" begin
            p = NegativeBinomial(3, 0.3)
            q = NegativeBinomial(3, 0.5)
            test_kl(p, q)
            @test kldivergence(p, q) ≈ 3 * kldivergence(Geometric(0.3), Geometric(0.5))
        end
        @testset "Normal" begin
            p = Normal(0, 1)
            q = Normal(0.5, 0.5)
            test_kl(p, q)
        end
        @testset "NormalCanon" begin
            p = NormalCanon(1, 2)
            q = NormalCanon(3, 4)
            test_kl(p, q)
            @test kldivergence(p, q) ≈ kldivergence(Normal(1/2, 1/sqrt(2)), Normal(3/4, 1/2))
        end
        @testset "Poisson" begin
            p = Poisson(4.0)
            q = Poisson(3.0)
            test_kl(p, q)

            # special case (test function also checks `kldivergence(p0, p0)`)
            p0 = Poisson(0.0)
            test_kl(p0, p)
        end
    end

    @testset "multivariate" begin
        @testset "AbstractMvNormal" begin
            p_mvnormal = MvNormal([0.2, -0.8], Diagonal([0.5, 0.75]))
            q_mvnormal = MvNormal([1.5, 0.5], Diagonal([1.0, 0.2]))
            test_kl(p_mvnormal, q_mvnormal)

            p_cholesky = CholeskyMvNormal([0.2, -0.8], Diagonal([0.5, 0.75]))
            q_cholesky = CholeskyMvNormal([1.5, 0.5], Diagonal([1.0, 0.2]))
            test_kl(p_cholesky, q_cholesky)

            # check consistency and mixed computations
            v = kldivergence(p_mvnormal, q_mvnormal)
            @test kldivergence(p_mvnormal, q_cholesky) ≈ v
            @test kldivergence(p_cholesky, q_mvnormal) ≈ v
            @test kldivergence(p_cholesky, q_cholesky) ≈ v
        end
    end
end
