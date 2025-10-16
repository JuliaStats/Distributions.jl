using Distributions
using ForwardDiff

using Test

@testset "TDist" begin
    @testset "Type stability of `rand` (#1614)" begin
        @inferred(rand(TDist(big"1.0")))
        @inferred(rand(TDist(ForwardDiff.Dual(1.0))))
    end

    for T in (Float32, Float64)
        @test @inferred(rand(TDist(T(1)))) isa T
    end
end
