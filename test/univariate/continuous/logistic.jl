using Distributions
using Test

@testset "Logistic" begin
    test_cgf(Logistic(0,   1), (-0.99,0.99, 1f-2, -1f-2))
    test_cgf(Logistic(100,10), (-0.099,0.099, 1f-2, -1f-2))

    # issue 1082
    @testset "rand consistency" begin
        for T in (Float32, Float64, BigFloat)
            @test @inferred(rand(Logistic(T(0), T(1)))) isa T
            @test @inferred(rand(Logistic(T(0), T(1)), 5)) isa Vector{T}
        end
    end
end
