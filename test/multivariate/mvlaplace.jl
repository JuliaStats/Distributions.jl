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

@testset "SymmetricMvLaplace constructor" begin
    mu = [1., 2., 3.]
    C = [4. -2. -1.; -2. 5. -1.; -1. -1. 6.]
    @test typeof(SymmetricMvLaplace(mu, PDMat(Array{Float32}(C)))) == typeof(SymmetricMvLaplace(mu, PDMat(C)))
    @test typeof(SymmetricMvLaplace(mu, Array{Float32}(C))) == typeof(SymmetricMvLaplace(mu, PDMat(C)))
    @test SymmetricMvLaplace(mu, I) === SymmetricMvLaplace(mu, Diagonal(Ones(length(mu))))
    @test SymmetricMvLaplace(mu, 9 * I) === SymmetricMvLaplace(mu, Diagonal(Fill(9, length(mu))))
    @test SymmetricMvLaplace(mu, 0.25f0 * I) === SymmetricMvLaplace(mu, Diagonal(Fill(0.25f0, length(mu))))
end