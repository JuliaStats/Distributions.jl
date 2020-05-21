using Distributions, PDMats
using Test, LinearAlgebra
import Distributions: ispossemdef

# RealInterval
r = RealInterval(1.5, 4.0)
@test minimum(r) == 1.5
@test maximum(r) == 4.0
@test extrema(r) == (1.5, 4.0)

@test partype(Gamma(1, 2)) == Float64
@test partype(Gamma(1.1, 2)) == Float64
@test partype(Normal(1//1, 2//1)) == Rational{Int}
@test partype(MvNormal(rand(Float32, 5), Matrix{Float32}(I, 5, 5))) == Float32

# special cases
@test partype(Kolmogorov()) == Float64
@test partype(Hypergeometric(2, 2, 2)) == Float64
@test partype(DiscreteUniform(0, 4)) == Float64

A = rand(1:10, 5, 5)
B = rand(Float32, 4)
C = 1//2
L = rand(Float32, 4, 4)
D = PDMats.PDMat(L * L')

# Ensure that utilities functions works with abstract arrays

@test isprobvec(GenericArray([1, 1, 1])) == false
@test isprobvec(GenericArray([1/3, 1/3, 1/3]))

# Positive definite matrix
M = GenericArray([1.0 0.0; 0.0 1.0])
# Non-invertible matrix
N = GenericArray([1.0 0.0; 1.0 0.0])

@test Distributions.isApproxSymmmetric(N) == false
@test Distributions.isApproxSymmmetric(M)


n = 10
areal = randn(n,n)/2
aimg  = randn(n,n)/2
@testset "For A containing $eltya" for eltya in (Float32, Float64, ComplexF32, ComplexF64, Int)
    ainit = eltya == Int ? rand(1:7, n, n) : convert(Matrix{eltya}, eltya <: Complex ? complex.(areal, aimg) : areal)
    @testset "Positive semi-definiteness" begin
        notsymmetric = ainit
        notsquare    = [ainit ainit]
        @test !ispossemdef(notsymmetric)
        @test !ispossemdef(notsquare)
        for truerank in 0:n
            X = ainit[:, 1:truerank]
            A = truerank == 0 ? zeros(eltya, n, n) : X * X'
            @test ispossemdef(A)
            for testrank in 0:n
                if testrank == truerank
                    @test ispossemdef(A, testrank)
                else
                    @test !ispossemdef(A, testrank)
                end
            end
            @test !ispossemdef(notsymmetric, truerank)
            @test !ispossemdef(notsquare, truerank)
            @test_throws ArgumentError ispossemdef(A, -1)
            @test_throws ArgumentError ispossemdef(A, n + 1)
        end
    end
end
