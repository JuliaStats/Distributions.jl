# Sampling Tests
@testset "NormalCanon sampling tests" begin
    for d in [
        NormalCanon()
        NormalCanon(-1.0, 2.5)
        NormalCanon(2.0, 0.8)
    ]
        test_distr(d, 10^6)
    end
end
