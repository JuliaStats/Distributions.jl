using Distributions
import PDMats: ScalMat, PDiagMat, PDMat
using LinearAlgebra, Test
using FillArrays


@testset "SymmetricMvLaplace tests" begin
    mu = [1., 2., 3.]
    mu_r = 1.:3.

    C = [4. -2. -1.; -2. 5. -1.; -1. -1. 6.]
    dv = [1.2, 3.4, 2.6]
    for (g, μ, Σ) in [
        (SymmetricMvLaplace(mu, C), mu, C),
        (SymmetricMvLaplace(mu_r, C), mu_r, C),
        (SymmetricMvLaplace(mu, Symmetric(C)), mu, Matrix(Symmetric(C))),
        (SymmetricMvLaplace(mu_r, Symmetric(C)), mu_r, Matrix(Symmetric(C))),
        (SymmetricMvLaplace(mu, Diagonal(dv)), mu, Matrix(Diagonal(dv))),
        (SymmetricMvLaplace(mu, Symmetric(Diagonal(dv))), mu, Matrix(Diagonal(dv))),
        (SymmetricMvLaplace(mu, Hermitian(Diagonal(dv))), mu, Matrix(Diagonal(dv))),
        (SymmetricMvLaplace(mu_r, Diagonal(dv)), mu_r, Matrix(Diagonal(dv))) ]

        @test mean(g)   ≈ μ
        @test cov(g)    ≈ Σ
        @test invcov(g) ≈ inv(Σ)
    end
end

@testset "SymmetricMvLaplace constructors" begin
    @testset "Providing mu and Sigma" begin
        mu = [1., 2., 3.]
        C = [4. -2. -1.; -2. 5. -1.; -1. -1. 6.]
        @test typeof(SymmetricMvLaplace(mu, PDMat(Array{Float32}(C)))) == typeof(SymmetricMvLaplace(mu, PDMat(C)))
        @test typeof(SymmetricMvLaplace(mu, Array{Float32}(C))) == typeof(SymmetricMvLaplace(mu, PDMat(C)))
        @test SymmetricMvLaplace(mu, I) === SymmetricMvLaplace(mu, Diagonal(Ones(length(mu))))
        @test SymmetricMvLaplace(mu, 9 * I) === SymmetricMvLaplace(mu, Diagonal(Fill(9, length(mu))))
        @test SymmetricMvLaplace(mu, 0.25f0 * I) === SymmetricMvLaplace(mu, Diagonal(Fill(0.25f0, length(mu))))
    end
    @testset "Providing mu, lambda and Gamma" begin
        mu = [1., 2., 3.]
        C = [4. -2. -1.; -2. 5. -1.; -1. -1. 6.]
        Csym = C'*C
        L = det(C)^(1/size(C,1))
        G = 1/L * C
        @test typeof(SymmetricMvLaplace(mu, L, PDMat(Array{Float32}(G)))) == typeof(SymmetricMvLaplace(mu, L, PDMat(G)))
        @test typeof(SymmetricMvLaplace(mu, L, Array{Float32}(G))) == typeof(SymmetricMvLaplace(mu, L, PDMat(G)))
        @test SymmetricMvLaplace(mu, 1, I) === SymmetricMvLaplace(mu, Diagonal(Ones(length(mu))))

        L9 = det(9*I(3))^(1/3)
        L025 = det(0.25f0*I(3))^(1/3)
        LC = det(Csym)^(1/3)
        @test SymmetricMvLaplace(mu, L9, I) === SymmetricMvLaplace(mu, L9, Diagonal(Fill(1, length(mu))))
        @test SymmetricMvLaplace(mu, L025, I) === SymmetricMvLaplace(mu, L025, Diagonal(Fill(1, length(mu))))
        @test SymmetricMvLaplace(mu, LC, 1/LC*Symmetric(Csym)) == SymmetricMvLaplace(mu, LC, 1/LC * Csym)
    end
end

@testset "Symmetric MvLaplace basic functions" begin
    for t in (Float32, Float64)
        @testset "Type: $t" begin
            mu = t.(ones(3))
            λ = t.(1)
            Γ = t.(I(3))
            d = SymmetricMvLaplace(mu, λ, Γ)
    
            @test length(d) == 3
            @test params(d) == (mu, λ, Γ)
            @test maximum(d) == [Inf, Inf, Inf]
            @test minimum(d) == [-Inf, -Inf, -Inf]
            @test mode(d) == mu
            @test modes(d) == [mu]
            @test var(d) == diag(λ*Γ)
            @test eltype(d) == t
            @test partype(d) == t
            @test insupport(d, rand(3))

            s = rand(d, 10)
            @test size(s) == (3, 10)
            @test eltype(s) == t
            @test isfinite(loglikelihood(d, s))

            l = Laplace(1, 0.707)
            s_d = rand(d, 10_000_000)
            qs = collect(0.01:0.01:0.99)
            q_d = quantile.(Ref(s_d[1,:]), qs)
            q_l = quantile.(Ref(l), qs)
            @test all(isapprox.(q_d, q_l; atol=1e-2)) ### tolerance a bit high as tails are heavy and a bit stochastic
            ### probably need to come up with some logpdf correctness tests too
        end
    end
end
