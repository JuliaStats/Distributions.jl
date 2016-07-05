# Test tail sampling of TruncatedNormal

using Distributions
using Base.Test

# tests tail sampling, #343, #493
@test isfinite(rand(TruncatedNormal(0,1,35,Inf)))
@test isfinite(rand(TruncatedNormal(0,1,-Inf,-35)))