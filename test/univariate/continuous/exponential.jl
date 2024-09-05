
@testset "Exponential" begin
    test_cgf(Exponential(1), (0.9, -1, -100f0, -1e6))
    test_cgf(Exponential(0.91), (0.9, -1, -100f0, -1e6))
    test_cgf(Exponential(10  ), (0.08, -1, -100f0, -1e6))

    for T in (Float32, Float64)
        @test @inferred(rand(Exponential(T(1)))) isa T
    end
end
