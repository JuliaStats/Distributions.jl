using Distributions, Random
using Test, LinearAlgebra, PDMats

#  dimensions

n = 6
p = 3

#  parameters

v    = 10
M    = randn(n, p)
Σ    = rand(InverseWishart(n + 2, Matrix(1.0I, n, n)))
Ω    = rand(InverseWishart(p + 2, Matrix(1.0I, p, p)))
ΣF32 = Matrix{Float32}(Σ)
ΩBF  = Matrix{BigFloat}(Ω)
PDΣ  = PDMat(Σ)
PDΩ  = PDMat(Ω)
μ    = randn(n)
θ    = randn(p)
σ²   = 2.0rand()

#  distribution instances

D = MatrixT(v, M, Σ, Ω)                                   #  n x p
L = MatrixT(v, reshape(μ, n, 1), Σ, reshape([σ²], 1, 1))  #  n x 1
H = MatrixT(v, reshape(θ, 1, p), reshape([σ²], 1, 1), Ω)  #  1 x p
K = MatrixT(1000, M, 1000Σ, Ω)

l  = MvTDist(L)
h  = MvTDist(H)
MN = MatrixNormal(M, Σ, Ω)

#  draws

A = rand(D)

X = rand(L)
Y = rand(H)

x = vec(X)
y = vec(Y)

@testset "Check all MatrixT constructors" begin
    @test MatrixT(v, M, Σ,        PDΩ.chol) isa MatrixT
    @test MatrixT(v, M, Σ,        PDΩ)      isa MatrixT
    @test MatrixT(v, M, PDΣ.chol, Ω)        isa MatrixT
    @test MatrixT(v, M, PDΣ.chol, PDΩ.chol) isa MatrixT
    @test MatrixT(v, M, PDΣ.chol, PDΩ)      isa MatrixT
    @test MatrixT(v, M, PDΣ,      Ω)        isa MatrixT
    @test MatrixT(v, M, PDΣ,      PDΩ.chol) isa MatrixT
    @test MatrixT(v, M, PDΣ,      PDΩ)      isa MatrixT
end

@testset "MatrixT promotion during construction" begin
    R = MatrixT(v, M, ΣF32, ΩBF)
    @test partype(R) == BigFloat
end

@testset "MatrixT construction errors" begin
    @test_throws ArgumentError MatrixT(-1, M, Σ, Ω)
    @test_throws ArgumentError MatrixT(v, M, Ω, Σ)
    @test_throws ArgumentError MatrixT(v, M, Ω, Ω)
    @test_throws ArgumentError MatrixT(v, M, Σ, Σ)
end

@testset "MatrixT params" begin
    vv, MM, PDΣΣ, PDΩΩ = params(D)
    @test v == vv
    @test M == MM
    @test Σ == PDΣΣ.mat
    @test Ω == PDΩΩ.mat
end

@testset "MatrixT size" begin
    @test size(D) == (n, p)
    @test size(L) == (n, 1)
    @test size(H) == (1, p)
end

@testset "MatrixT rank" begin
    @test rank(D) == min(n, p)
    @test rank(L) == 1
    @test rank(H) == 1
    @test rank(D) == rank(rand(D))
    @test rank(L) == rank(rand(L))
    @test rank(H) == rank(rand(H))
end

@testset "MatrixT insupport" begin
    @test insupport(D, rand(D))
    @test insupport(L, rand(L))
    @test insupport(H, rand(H))

    @test !insupport(D, Matrix(rand(D)'))
    @test !insupport(D, M + M * im)
    @test !insupport(L, rand(H))
    @test !insupport(H, rand(L))
end

@testset "MatrixT mean" begin
    @test mean(D) == M
    @test mean(L) == reshape(μ, n, 1)
    @test mean(H) == reshape(θ, 1, p)

    @test_throws ArgumentError mean( MatrixT(1, M, Σ, Ω) )
end

@testset "MatrixT mode" begin
    @test mode(D) == M
    @test mode(L) == reshape(μ, n, 1)
    @test mode(H) == reshape(θ, 1, p)
end

@testset "MatrixT logpdf" begin
    #  Check against MvTDist
    @test logpdf(L, X) ≈ logpdf(l, x)
    @test logpdf(H, Y) ≈ logpdf(h, y)

    #  MT(v, M, vΣ, Ω) -> MN(M, Σ, Ω) as v -> ∞
    @test isapprox(logpdf(K, A), logpdf(MN, A), atol = 0.1)
end

@testset "MatrixT sample moments" begin
    @test isapprox(mean(rand(D, 100000)), mean(D) , atol = 0.1)
    @test isapprox(cov(hcat(vec.(transpose.(rand(D, 100000)))...)'), kron(Σ, Ω) ./ (v - 2), atol = 0.1)
end

@testset "MatrixT conversion" for elty in (Float32, Float64, BigFloat)
    Del1 = convert(MatrixT{elty}, D)
    Del2 = convert(MatrixT{elty}, v, M, PDΣ, PDΩ, D.c0)

    @test partype(Del1) == elty
    @test partype(Del2) == elty
end
