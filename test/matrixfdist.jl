using Distributions, Random
using Test, LinearAlgebra, PDMats

p  = 3
n1 = p + abs(10randn())
n2 = p + 1 + abs(10randn())
B  = rand( InverseWishart(p + 2, Matrix(1.0I, p, p)) )
b  = Matrix((n2 / n1) * I, 1, 1)

G = MatrixFDist(n1, n2, B)
F = MatrixFDist(n1, n2, b)
f = FDist(n1, n2)

@testset "MatrixFDist promotion during construction" begin
    R = MatrixFDist(Float32(n1), BigFloat(n2), Matrix{Float64}(B))
    @test partype(R) == BigFloat
end

@testset "MatrixFDist construction errors" begin
    @test_throws ArgumentError MatrixFDist(1, n2, B)
    @test_throws ArgumentError MatrixFDist(n1, 1, B)
    @test_throws ArgumentError mean(MatrixFDist(n1, p, B))
end

@testset "MatrixFDist params" begin
    df1, df2, PDB = params(G)
    @test df1 == n1
    @test df2 == n2
    @test PDB.mat == B
end

@testset "MatrixFDist dim" begin
    @test dim(G) == p
    @test dim(F) == 1
end

@testset "MatrixFDist size" begin
    @test size(G) == (p, p)
    @test size(F) == (1, 1)
end

@testset "MatrixFDist rank" begin
    @test rank(G) == p
    @test rank(F) == 1
    @test rank(G) == rank(rand(G))
    @test rank(F) == rank(rand(F))
end

@testset "MatrixFDist insupport" begin
    @test insupport(G, rand(G))
    @test insupport(F, rand(F))

    @test !insupport(G, rand(G) + rand(G) * im)
    @test !insupport(G, randn(p, p + 1))
    @test !insupport(G, randn(p, p))
end

@testset "MatrixFDist logpdf" begin
    Z = rand(F)
    z = Z[1, 1]
    @test logpdf(F, Z) ≈ logpdf(f, z)
end

@testset "MatrixFDist sample moments" begin
    @test isapprox(mean(rand(G, 100000)), mean(G) , atol = 0.1)
end

@testset "MatrixFDist conversion" for elty in (Float32, Float64, BigFloat)
    Gel1 = convert(MatrixFDist{elty}, G)
    Gel2 = convert(MatrixFDist{elty}, G.W, n2, G.logc0)

    @test Gel1 isa MatrixFDist{elty, Wishart{elty, PDMat{elty,Array{elty,2}}}}
    @test Gel2 isa MatrixFDist{elty, Wishart{elty, PDMat{elty,Array{elty,2}}}}
    @test partype(Gel1) == elty
    @test partype(Gel2) == elty
end
