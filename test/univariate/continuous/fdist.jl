using Distributions, Test

@testset "FDist" begin
    @testset "kurtosis" begin
        d = FDist(30, 40)
        @test @inferred(kurtosis(d)) ≈ 2.2442906574394463
    end
end
