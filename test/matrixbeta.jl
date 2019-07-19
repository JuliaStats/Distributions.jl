using Distributions, Random
using Test, LinearAlgebra, PDMats

p = 4
n1 = p + abs(10randn())
n2 = p + abs(10randn())

B = MatrixBeta(p, n1, n2)
C = MatrixBeta(1, n1, n2)
c = Beta(n1/2, n2/2)

Z = rand(C)
z = Z[1]

@testset "MatrixBeta promotion during construction" begin
    R = MatrixBeta(p, Float32(n1), BigFloat(n2))
    @test partype(R) == BigFloat
end

@testset "MatrixBeta construction errors" begin
    @test_throws ArgumentError MatrixBeta(-2, n1, n2)
    @test_throws ErrorException MatrixBeta(p, p - 2, n2)
    @test_throws ErrorException MatrixBeta(p, n1, p - 2)
end

@testset "MatrixBeta params" begin
    df1, df2 = params(B)
    @test df1 == n1
    @test df2 == n2
end

@testset "MatrixBeta dim" begin
    @test dim(B) == p
    @test dim(C) == 1
end

@testset "MatrixBeta size" begin
    @test size(B) == (p, p)
    @test size(C) == (1, 1)
end

@testset "MatrixBeta rank" begin
    @test rank(B) == p
    @test rank(C) == 1
    @test rank(B) == rank(rand(B))
    @test rank(C) == rank(rand(C))
end

@testset "MatrixBeta insupport" begin
    @test insupport(B, rand(B))
    @test insupport(C, rand(C))
    @test all(0 .< eigvals(rand(B)) .< 1)

    @test !insupport(B, rand(B) + rand(B) * im)  #  not real
    @test !insupport(B, rand(C))                 #  wrong dims
    @test !insupport(C, rand(B))                 #  wrong dims
    @test !insupport(B, rand(p, p))              #  U isn't posdef
    @test !insupport(B, rand(B) + I)             #  I - U isn't posdef
end

@testset "MatrixBeta logpdf" begin
    @test logpdf(C, Z) ≈ logpdf(c, z)
end

@testset "MatrixBeta sample moments" begin
    @test isapprox(mean(rand(B, 10000)), mean(B) , atol = 0.1)
    @test isapprox(mean(rand(C, 10000))[1, 1], mean(c) , atol = 0.1)
end

@testset "MatrixBeta conversion" for elty in (Float32, Float64, BigFloat)
    Bel1 = convert(MatrixBeta{elty}, B)
    Bel2 = convert(MatrixBeta{elty}, B.W1, B.W2, B.logc0)

    @test partype(Bel1) == elty
    @test partype(Bel2) == elty
end
