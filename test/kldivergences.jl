@testset "KL divergences" begin
    function logdiff(P, Q, x)
        logp = logpdf(P, x)
        return (logp > -Inf) * (logp - logpdf(Q, x))
    end
    function test_kl(p, q)
        @test kldivergence(p, q) > 0
        @test kldivergence(p, p) ≈ 0 atol=1e-1
        @test kldivergence(q, q) ≈ 0 atol=1e-1
        @test kldivergence(p, q) ≈ Distributions.mcexpectation(Random.GLOBAL_RNG, x -> logdiff(p, q, x), sampler(p), 10000) atol=1e-1
    end
    @testset "univariate" begin
        @testset "Beta" begin
            p = Beta(2, 10)
            q = Beta(3, 5)
            test_kl(p, q)
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
        @testset "InverseGamma" begin
            p = InverseGamma(2.0, 1.0)
            q = InverseGamma(3.0, 2.0)
            test_kl(p, q)
        end
        @testset "Normal" begin
            p = Normal(0, 1)
            q = Normal(0.5, 0.5)
            test_kl(p, q)
        end
        @testset "Poisson" begin
            p = Poisson(4.0)
            q = Poisson(3.0)
            test_kl(p, q)
        end
    end
    @testset "multivariate" begin
        # We try on a newly created AbstractMvNormal
        struct CholeskyMvNormal{TL,Tm} <: Distributions.AbstractMvNormal
            m::Tm
            L::TL
        end
        Distributions.mean(p::CholeskyMvNormal) = p.m
        Distributions.cov(p::CholeskyMvNormal) = p.L * p.L'
        Distributions.logdetcov(p::CholeskyMvNormal) = 2 * sum(log, diag(p.L))
        Distributions.sqmahal(p::CholeskyMvNormal, x::AbstractVector) = sum(abs2, p.L \ (mean(p) - x))
        Distributions._rand!(rng::AbstractRNG, p::CholeskyMvNormal, x::Vector) = x .= p.m .+ p.L * randn!(rng, x) 
        Distributions.length(p::CholeskyMvNormal) = length(p.m)
        function Distributions.logpdf(p::CholeskyMvNormal, x::AbstractVector)
            return -0.5 * (length(p) * log(2π) + 2 * logdet(p.L) + sum(abs2, p.L \ (x .- p.m)))
        end
        @testset "AbstractMvNormal" begin
            n_dim = 2
            X1 = cholesky(Matrix(0.5 * I(n_dim))).L
            X2 = cholesky(Matrix(0.3 * I(n_dim))).L
            p = CholeskyMvNormal(zeros(n_dim), X1)
            q = CholeskyMvNormal(ones(n_dim), X2)
            test_kl(p, q)
        end
        @testset "MvNormal" begin
            n_dim = 2
            p = MvNormal(zeros(n_dim), Matrix(0.5 * I(n_dim)))
            q = MvNormal(ones(n_dim), Matrix(0.3 * I(n_dim)))
            test_kl(p, q)
        end
    end

end
