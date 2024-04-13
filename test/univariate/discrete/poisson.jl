using Test, Distributions, OffsetArrays

test_cgf(Poisson(1   ), (1f0,2f0,10.0,50.0))
test_cgf(Poisson(10  ), (1f0,2f0,10.0,50.0))
test_cgf(Poisson(1e-3), (1f0,2f0,10.0,50.0))

@testset "Poisson suffstats and OffsetArrays" begin
    a = rand(Poisson(), 11)
    wa = 1.0:11.0

    resulta = @inferred(suffstats(Poisson, a))

    resultwa = @inferred(suffstats(Poisson, a, wa))

    b = OffsetArray(a, -5:5)
    wb = OffsetArray(wa, -5:5)

    resultb = @inferred(suffstats(Poisson, b))
    @test resulta == resultb

    resultwb = @inferred(suffstats(Poisson, b, wb))
    @test resultwa == resultwb

    @test_throws DimensionMismatch suffstats(Poisson, a, wb)
end