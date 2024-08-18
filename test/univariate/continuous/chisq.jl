
@testset "Chisq" begin
    test_cgf(Chisq(1), (0.49, -1, -100, -1.0f6))
    test_cgf(Chisq(3), (0.49, -1, -100, -1.0f6))
end
