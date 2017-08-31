# Test for OffsetArrays
using Distributions
using OffsetArrays
using Base.Test

d = Normal(0,1)
v = [-1.0, 0.0, 1.0]
r = -1:1
@test pdf(d, OffsetArray(v,r)) == OffsetArray(pdf(d, v), r)
