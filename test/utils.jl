using Distributions
using Base.Test

# RealInterval
r = RealInterval(1.5, 4.0)
@test minimum(r) == 1.5
@test maximum(r) == 4.0

@test partype(Gamma(1, 2)) == Float64
@test partype(Gamma(1.1, 2)) == Float64
@test partype(Normal(1//1, 2//1)) == Rational{Int64}
@test partype(MvNormal(rand(Float32, 5), eye(Float32, 5))) == Float32

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
@test Distributions.promote_eltype(A, B) == (convert(Array{Float32}, A), convert(Array{Float32}, B))
@test Distributions.promote_eltype(A, C) == (convert(Array{Rational{Int}}, A), convert(Float32, C))
AA, DD = Distributions.promote_eltype(A, D)
@test AA == convert(Array{Float32}, A)
@test DD.mat == convert(PDMats.PDMat{Float32}, D).mat
ZZ, DD = Distributions.promote_eltype(Z, D)
@test ZZ == Distributions.ZeroVector{Float32}(5)
@test DD.mat == convert(PDMats.PDMat{Float32}, D).mat
@test Distributions.convert_eltype(Float32, A) == convert(Array{Float32}, A)
@test Distributions.convert_eltype(Float32, Z) == Distributions.ZeroVector{Float32}(length(Z))
@test Distributions.convert_eltype(Float64, D).mat == convert(PDMats.PDMat{Float64}, D).mat
@test typeof(convert(Distributions.ZeroVector{Float32}, Z)) == Distributions.ZeroVector{Float32}
