using Distributions, Random
using Test, LinearAlgebra, PDMats

#  dimensions

n = 3
p = 6

#  parameters

M    = randn(n, p)
U    = rand(InverseWishart(n + 2, Matrix(1.0I, n, n)))
V    = rand(InverseWishart(p + 2, Matrix(1.0I, p, p)))
UF32 = Matrix{Float32}(U)
VBF  = Matrix{BigFloat}(V)
PDU  = PDMat(U)
PDV  = PDMat(V)
m    = randn(n)
w    = randn(p)
μ    = randn()
σ    = 2.0rand()
σ²   = σ ^ 2

#  distribution instances

D = MatrixNormal(M, U, V)  #  n x p
G = MatrixNormal(p, n)     #  p x n
L = MatrixNormal(reshape(m, n, 1), U, reshape([σ²], 1, 1))  #  n x 1
H = MatrixNormal(reshape(w, 1, p), reshape([σ²], 1, 1), V)  #  1 x p
K = MatrixNormal(reshape([μ], 1, 1), reshape([σ], 1, 1), reshape([σ], 1, 1))  #  1 x 1

d = vec(D) # MvNormal(vec(M), V ⊗ U)
g = MvNormal( Matrix(1.0I, p*n, p*n) )
l = MvNormal(m, σ² * U)
h = MvNormal(w, σ² * V)
k = Normal(μ, σ)

#  draws

A = rand(D)
B = rand(G)
X = rand(L)
Y = rand(H)
Z = rand(K)

a = vec(A)
b = vec(B)
x = vec(X)
y = vec(Y)
z = Z[1, 1]

@testset "Check all MatrixNormal constructors" begin
    @test MatrixNormal(M, U,        PDV.chol) isa MatrixNormal
    @test MatrixNormal(M, U,        PDV)      isa MatrixNormal
    @test MatrixNormal(M, PDU.chol, V)        isa MatrixNormal
    @test MatrixNormal(M, PDU.chol, PDV.chol) isa MatrixNormal
    @test MatrixNormal(M, PDU.chol, PDV)      isa MatrixNormal
    @test MatrixNormal(M, PDU,      V)        isa MatrixNormal
    @test MatrixNormal(M, PDU,      PDV.chol) isa MatrixNormal
    @test MatrixNormal(M, PDU,      PDV)      isa MatrixNormal
end

@testset "MatrixNormal promotion during construction" begin
    R = MatrixNormal(M, UF32, VBF)
    @test partype(R) == BigFloat
end

@testset "MatrixNormal construction errors" begin
    @test_throws ArgumentError MatrixNormal(M, V, U)
    @test_throws ArgumentError MatrixNormal(M, U, U)
end

@testset "MatrixNormal params" begin
    MM, PDUU, PDVV = params(D)
    @test MM == M
    @test U == PDUU.mat
    @test V == PDVV.mat
end

@testset "MatrixNormal size" begin
    @test size(D) == (n, p)
    @test size(G) == (p, n)
    @test size(L) == (n, 1)
    @test size(H) == (1, p)
    @test size(K) == (1, 1)
end

@testset "MatrixNormal rank" begin
    @test rank(D) == min(n, p)
    @test rank(G) == min(n, p)
    @test rank(L) == 1
    @test rank(H) == 1
    @test rank(K) == 1

    @test rank(D) == rank(rand(D))
    @test rank(G) == rank(rand(G))
    @test rank(L) == rank(rand(L))
    @test rank(H) == rank(rand(H))
    @test rank(K) == rank(rand(K))
end

@testset "MatrixNormal insupport" begin
    @test insupport(D, rand(D))
    @test insupport(G, rand(G))
    @test insupport(L, rand(L))
    @test insupport(H, rand(H))
    @test insupport(K, rand(K))

    @test !insupport(D, rand(G))
    @test !insupport(D, M + M * im)
    @test !insupport(L, rand(H))
    @test !insupport(H, rand(L))
end

@testset "MatrixNormal mean" begin
    @test mean(D) == M
    @test mean(G) == zeros(p, n)
    @test mean(L) == reshape(m, n, 1)
    @test mean(H) == reshape(w, 1, p)
    @test mean(K) == reshape([μ], 1, 1)
end

@testset "MatrixNormal mode" begin
    @test mode(D) == M
    @test mode(G) == zeros(p, n)
    @test mode(L) == reshape(m, n, 1)
    @test mode(H) == reshape(w, 1, p)
    @test mode(K) == reshape([μ], 1, 1)
end

@testset "MatrixNormal logpdf" begin
    @test logpdf(D, A) ≈ logpdf(d, a)
    @test logpdf(G, B) ≈ logpdf(g, b)
    @test logpdf(L, X) ≈ logpdf(l, x)
    @test logpdf(H, Y) ≈ logpdf(h, y)
    @test logpdf(K, Z) ≈ logpdf(k, z)
end

@testset "MatrixNormal sample moments" begin
    @test isapprox(mean(rand(D, 100000)), mean(D) , atol = 0.1)
    @test isapprox(cov(hcat(vec.(rand(D, 100000))...)'), kron(V, U) , atol = 0.1)
end

@testset "MatrixNormal conversion" for elty in (Float32, Float64, BigFloat)
    Del1 = convert(MatrixNormal{elty}, D)
    Del2 = convert(MatrixNormal{elty}, M, PDU, PDV, D.logc0)

    @test partype(Del1) == elty
    @test partype(Del2) == elty
end
