using Distributions
using ForwardDiff

using Test

@testset "TDist" begin
    @testset "Type stability of `rand` (#1614)" begin
        if VERSION >= v"1.9.0-DEV.348"
            # randn(::BigFloat) was only added in https://github.com/JuliaLang/julia/pull/44714
            @inferred(rand(TDist(big"1.0")))
        end
        @inferred(rand(TDist(ForwardDiff.Dual(1.0))))

    end

    @test rand(TDist(1.0)) isa Float64
    @test rand(TDist(1.0f0)) isa Float32

    @test entropy(TDist(1.0)) isa Float64
    @test entropy(TDist(1.0f0)) isa Float32
end
