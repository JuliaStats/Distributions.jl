using Distributions
using ForwardDiff

using Test

@testset "Type stability of `rand` (#1614)" begin
    @inferred(rand(TDist(big"1.0")))
    @inferred(rand(TDist(ForwardDiff.Dual(1.0))))
end
