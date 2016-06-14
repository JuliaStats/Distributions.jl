using Distributions
using Base.Test

# RealInterval
r = RealInterval(1.5, 4.0)
@test minimum(r) == 1.5
@test maximum(r) == 4.0

@test partype(Gamma(1, 2)) == Int64
@test partype(Gamma(1.1, 2)) == Float64
@test partype(MvNormal(rand(Float16, 5), eye(Float32, 5))) == Float32

A = rand(Int, 5, 5)
B = rand(Float32, 4)
C = 1//2
L = rand(Float16, 4, 4)
D = PDMat(L * L')
@test promote_eltype(A, B) == (convert(Array{Float32}, A), convert(Array{Float32}, B))
@test promote_eltype(A, C) == (convert(Array{Float32}, A), convert(Float32, C))
@test promote_eltype(A, D) == (A, convert(PDMat{Float32}, D))
