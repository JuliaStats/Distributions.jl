
@testset "Chisq" begin
    test_cgf(Chisq(1), (0.49, -1, -100, -1.0f6))
    test_cgf(Chisq(3), (0.49, -1, -100, -1.0f6))

    for T in (Float32, Float64)
        @test @inferred(rand(Chisq(T(1)))) isa T
    end
end
