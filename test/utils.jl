using Distributions
using Base.Test

# RealInterval
r = RealInterval(1.5, 4.0)
@test minimum(r) == 1.5
@test maximum(r) == 4.0

@test partype(Gamma(1, 2)) == Int64
@test partype(Gamma(1.1, 2)) == Float64
@test partype(MvNormal(rand(Float16, 5), eye(Float32, 5))) == Float32
