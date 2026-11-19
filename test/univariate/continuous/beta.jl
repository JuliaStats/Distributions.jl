using Distributions
using Test

@testset "beta.jl" begin
    # issue #1907
    @testset "rand consistency" begin
        for T in (Float32, Float64)
            @test @inferred(rand(Beta(T(1), T(1)))) isa T
            @test @inferred(rand(Beta(T(4//5), T(4//5)))) isa T
            @test @inferred(rand(Beta(T(1), T(2)))) isa T
            @test @inferred(rand(Beta(T(2), T(1)))) isa T

            @test @inferred(eltype(rand(Beta(T(1), T(1)), 2))) === T
            @test @inferred(eltype(rand(Beta(T(4//5), T(4//5)), 2))) === T
            @test @inferred(eltype(rand(Beta(T(1), T(2)), 2))) === T
            @test @inferred(eltype(rand(Beta(T(2), T(1)), 2))) === T
        end
    end
end
