@testset "Cosine" begin
    @testset "mgf, cgf, cf" begin
        @test cf(Cosine(1, 1), 1) ≈ cis(1) * sin(1)/(1 - π^-2)
        @test mgf(Cosine(-1, 2), -1) ≈ exp(1) * sinh(-2) / (-2*(1 + 4π^-2))
        @test cgf(Cosine(-1, 2), 2) ≈ -2 + log(sinh(4)) - log(4) - log(1 + 16π^-2)
    end

    @testset "affine transformations" begin
        @test -Cosine(1, 1) == Cosine(-1, 1)
        @test 3Cosine(1, 2) == Cosine(3, 6)
        @test Cosine(0, 2) + 42 == Cosine(42, 2)
    end
end
