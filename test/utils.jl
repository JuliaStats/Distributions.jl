using Distributions, PDMats
using Test, LinearAlgebra


# RealInterval
r = RealInterval(1.5, 4.0)
@test minimum(r) == 1.5
@test maximum(r) == 4.0

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
Z = Distributions.ZeroVector(Float64, 5)
L = rand(Float32, 4, 4)
D = PDMats.PDMat(L * L')
@test typeof(convert(Distributions.ZeroVector{Float32}, Z)) == Distributions.ZeroVector{Float32}
