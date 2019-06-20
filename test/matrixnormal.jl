using Distributions, Random
using Test, LinearAlgebra, PDMats

n = 3
p = 6

M = randn(n, p)
U = rand(InverseWishart(n + 2, Matrix(1.0I, n, n)))
V = rand(InverseWishart(p + 2, Matrix(1.0I, p, p)))

D = MatrixNormal(M, U, V)
G = MatrixNormal(p, n)
d = vec(D) # MvNormal(vec(M), V ⊗ U)

MM, PDU, PDV = params(D)
X = rand(D)
x = vec(X)
y = rand(d)
Y = reshape(y, n, p)

MM, PDU, PDV = params(D)

@test size(D) == (n, p)
@test rank(D) == min(n, p)
@test insupport(D, rand(D))
@test !insupport(D, rand(G))
@test !insupport(D, M + M*im)
@test mean(D) == D.M
@test mean(G) == zeros(p, n)
@test mode(D) == D.M
@test mode(G) == zeros(p, n)
@test MM == M
@test PDU.chol == PDMat(U).chol
@test PDV.chol == PDMat(V).chol
@test partype(D) == Float64
@test logpdf(d, x) ≈ logpdf(D, X)
@test logpdf(d, y) ≈ logpdf(D, Y)
@test isapprox(mean(rand(D, 100000)), mean(D) , atol=0.1)
