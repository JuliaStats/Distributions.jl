
@testset "Chisq" begin
    test_cgf(Chisq(1), (0.49, -1, -100, -1.0f6))
    test_cgf(Chisq(3), (0.49, -1, -100, -1.0f6))

    @test rand(Chisq(1.0)) isa Float64
    @test rand(Chisq(1.0f0)) isa Float32
end
