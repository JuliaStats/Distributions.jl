using Distributions, Random
using Test, LinearAlgebra, PDMats

#  dimensions

m = 3
n = 6

# parameters

M    = randn(m, n)
U    = rand(InverseWishart(m + 2, Matrix(1.0I, m, m)))
V    = rand(InverseWishart(n + 2, Matrix(1.0I, n, n)))
Σ    = Matrix(kron(V, U))
Σ32 = Matrix{Float32}(Σ)
ΣBF  = Matrix{BigFloat}(Σ)
PDΣ  = PDMat(Σ)
ml   = randn(m)
w    = randn(n)
μ    = randn()
σ    = 2.0rand()
σ²   = σ ^ 2


D = MatrixGaussian(M, Σ)  #  m x n
G = MatrixGaussian(n, m)     #  n x m
L = MatrixGaussian(reshape(ml,m, 1), U)  #  m x 1
H = MatrixGaussian(reshape(w, 1, n), V)  #  1 x n
K = MatrixGaussian(reshape([μ], 1, 1), reshape([σ²], 1, 1))  #  1 x 1

d = vec(D) # MvNormal(vec(M), V ⊗ U)
g = MvNormal( Matrix(1.0I, m*n, m*n) )
l = MvNormal(ml, U)
h = MvNormal(w, V)
k = Normal(μ, σ)


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

@testset "Check all MatrixGaussian constructors" begin
    @test MatrixGaussian(M, Σ) isa MatrixGaussian
    @test MatrixGaussian(M, PDΣ) isa MatrixGaussian
    @test MatrixGaussian(M, PDΣ.chol) isa MatrixGaussian
end


@testset "MatrixGaussian construction errors" begin
    @test_throws ArgumentError MatrixGaussian(M, [1 0; 0 1.0])
end

@testset "MatrixGaussian params" begin
    mm, nn, MM, PDΣΣ = params(D)
    @test m == mm
    @test n == nn
    @test M == reshape(MM, m, n)
    @test Σ == PDΣΣ.mat
end

@testset "MatrixGaussian size" begin
    @test size(D) == (m, n)
    @test size(G) == (n, m)
    @test size(L) == (m, 1)
    @test size(H) == (1, n)
    @test size(K) == (1, 1)
end

@testset "MatrixGaussian rank" begin
    @test rank(D) == min(m, n)
    @test rank(G) == min(m, n)
    @test rank(L) == 1
    @test rank(H) == 1
    @test rank(K) == 1

    @test rank(D) == rank(rand(D))
    @test rank(G) == rank(rand(G))
    @test rank(L) == rank(rand(L))
    @test rank(H) == rank(rand(H))
    @test rank(K) == rank(rand(K))
end

@testset "MatrixGaussian insupport" begin
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

@testset "MatrixGaussian mean" begin
    @test mean(D) == M
    @test mean(G) == zeros(n, m)
    @test mean(L) == reshape(ml, m, 1)
    @test mean(H) == reshape(w, 1, n)
    @test mean(K) == reshape([μ], 1, 1)
end

@testset "MatrixGaussian mode" begin
    @test mode(D) == M
    @test mode(G) == zeros(n, m)
    @test mode(L) == reshape(ml, m, 1)
    @test mode(H) == reshape(w, 1, n)
    @test mode(K) == reshape([μ], 1, 1)
end

@testset "MatrixGaussian cov and var" begin
    @test vec(var(D)) ≈ diag(cov(D))
    @test cov(D) ≈ cov(d)
    @test cov(G) ≈ cov(g)
    @test cov(L) ≈ cov(l)
    @test cov(H) ≈ cov(h)
    @test var(K)[1] ≈ var(k)
end

@testset "MatrixGaussian logpdf" begin
    @test logpdf(D, A) ≈ logpdf(d, a)
    @test logpdf(G, B) ≈ logpdf(g, b)
    @test logpdf(L, X) ≈ logpdf(l, x)
    @test logpdf(H, Y) ≈ logpdf(h, y)
    @test logpdf(K, Z) ≈ logpdf(k, z)
end

@testset "MatrixGaussian sample moments" begin
    @test isapprox(mean(rand(D, 100000)), mean(D) , atol = 0.1)
    @test isapprox(cov(hcat(vec.(rand(D, 100000))...)'), cov(D) , atol = 0.1)
end


@testset "PDMat mixing and matching" begin
    n = 3
    p = 4

    M = randn(n, p)

    u = rand()
    U_scale = ScalMat(n*p, u)
    U_dense = Matrix(U_scale)
    U_pd    = PDMat(U_dense)
    U_pdiag = PDiagMat(u*ones(n*p))


    baseeval = logpdf(MatrixGaussian(M, U_dense), M)

    for U in [U_scale, U_dense, U_pd, U_pdiag]
            d = MatrixGaussian(M, U)
            @test cov(d) ≈ Matrix(U)
            @test logpdf(d, M) ≈ baseeval
    end
end
