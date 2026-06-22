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
@test partype(Hypergeometric(2, 2, 2)) == Int
@test partype(Hypergeometric(2.0, 2, 2)) == Int
@test partype(DiscreteUniform(0, 4)) == Int
@test partype(DiscreteUniform(0.0, 4)) == Int

# `partype` is defined on the type; the instance method forwards for convenience
@test partype(Normal{Float32}) === Float32
@test partype(Normal(1.0f0, 2.0f0)) === partype(Normal{Float32}) === Float32
@test partype(Gamma{Float64}) === Float64
@test partype(DiscreteUniform) === Int
@test partype(Hypergeometric) === Int

# wrapper distributions recover the parameter type from the wrapped type parameter
@test partype(Truncated{Normal{Float64},Continuous,Float64}) === Float64
@test partype(truncated(Normal(0.0f0, 1.0f0), 0, 1)) === Float32
@test partype(censored(Normal(0.0f0, 1.0f0); lower = 0)) === Float32

# parametric distributions that previously fell back to the `Float64` default
# now report their actual parameter type
@test partype(Dirac{Float32}) === Float32
@test partype(Dirac(1.0f0)) === Float32
@test partype(DiscreteNonParametric([1, 2], Float32[0.5, 0.5])) === Float32

# distributions without parameters have an empty parameter promotion (`Union{}`),
# which is the identity for `promote_type` and composes through wrappers
@test partype(Chernoff()) === Union{}
@test partype(Kolmogorov()) === Union{}
@test partype(truncated(Kolmogorov(); lower = 0.0, upper = 1.0)) === Float64
# the sample size `n` is the only parameter of the Kolmogorov-Smirnov distributions
@test partype(KSDist(5)) === Int
@test partype(KSOneSided(5)) === Int

# the generic default for an unknown distribution type: parameters are `<:Real`, and
# `eltype(Real) === Real`, so the fallback is `Real` (and `zero`/`one` work on it)
@test partype(Distribution) === Real

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
