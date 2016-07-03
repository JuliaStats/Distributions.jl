using Distributions
using Base.Test

# RealInterval
r = RealInterval(1.5, 4.0)
@test minimum(r) == 1.5
@test maximum(r) == 4.0

@test partype(Gamma(1, 2)) == Int64
@test partype(Gamma(1.1, 2)) == Float64
@test partype(MvNormal(rand(Float32, 5), eye(Float32, 5))) == Float32

A = rand(1:10, 5, 5)
B = rand(Float32, 4)
C = 1//2
L = rand(Float32, 4, 4)
D = PDMats.PDMat(L * L')
@test Distributions.promote_eltype(A, B) == (convert(Array{Float32}, A), convert(Array{Float32}, B))
@test Distributions.promote_eltype(A, C) == (convert(Array{Rational{Int}}, A), convert(Float32, C))
AA, DD = Distributions.promote_eltype(A, D)
@test AA == convert(Array{Float32}, A)
@test DD.mat == convert(PDMats.PDMat{Float32}, D).mat
