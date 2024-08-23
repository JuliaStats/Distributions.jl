using Test, Distributions, OffsetArrays

@testset "Gamma" begin
    test_cgf(Gamma(1, 1), (0.9, -1, -100.0f0, -1e6))
    test_cgf(Gamma(10, 1), (0.9, -1, -100.0f0, -1e6))
    test_cgf(Gamma(0.2, 10), (0.08, -1, -100.0f0, -1e6))

    @testset "Gamma suffstats and OffsetArrays" begin
        a = rand(Gamma(), 11)
        wa = 1.0:11.0

        resulta = @inferred(suffstats(Gamma, a))

        resultwa = @inferred(suffstats(Gamma, a, wa))

        b = OffsetArray(a, -5:5)
        wb = OffsetArray(wa, -5:5)

        resultb = @inferred(suffstats(Gamma, b))
        @test resulta == resultb

        resultwb = @inferred(suffstats(Gamma, b, wb))
        @test resultwa == resultwb

        @test_throws DimensionMismatch suffstats(Gamma, a, wb)
    end

    for T in (Float32, Float64)
        @test @inferred(rand(Gamma(T(1), T(1)))) isa T
        @test @inferred(rand(Gamma(1/T(2), T(1)))) isa T
        @test @inferred(rand(Gamma(T(2), T(1)))) isa T
    end
end
