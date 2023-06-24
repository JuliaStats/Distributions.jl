using Test, Distributions, OffsetArrays

test_cgf(Gamma(1  ,1  ), (0.9, -1, -100f0, -1e6))
test_cgf(Gamma(10 ,1  ), (0.9, -1, -100f0, -1e6))
test_cgf(Gamma(0.2, 10), (0.08, -1, -100f0, -1e6))

@testset "Gamma suffstats and OffsetArrays" begin
    a = [ 2.0473208147256705, 17.82001658288441, 11.151923515093264, 5.577346651637625, 3.9024607405053313,
        13.3630618949226, 8.626021369961277, 8.987097909704644, 9.901974166860912, 7.317596579736286,
        1.5115770973132754]     # data generated randomly from Gamma in Julia
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
