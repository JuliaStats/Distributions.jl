@testset "Exponential" begin
    test_cgf(Exponential(1), (0.9, -1, -100f0, -1e6))
    test_cgf(Exponential(0.91), (0.9, -1, -100f0, -1e6))
    test_cgf(Exponential(10  ), (0.08, -1, -100f0, -1e6))

    for T in (Float32, Float64)
        @test @inferred(rand(Exponential(T(1)))) isa T
    end
end

test_cgf(Exponential(1), (0.9, -1, -100f0, -1e6))
test_cgf(Exponential(0.91), (0.9, -1, -100f0, -1e6))
test_cgf(Exponential(10), (0.08, -1, -100f0, -1e6))

# Sampling Tests
@testset "Exponential sampling tests" begin
    for d in [
        Exponential(1),
        Exponential(0.91),
        Exponential(10)
    ]
        test_distr(d, 10^6)
    end
end
