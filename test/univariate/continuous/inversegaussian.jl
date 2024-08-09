# Sampling Tests
@testset "InverseGaussian sampling tests" begin
    for d in [
        InverseGaussian()
        InverseGaussian(0.8)
        InverseGaussian(2.0)
        InverseGaussian(1.0, 1.0)
        InverseGaussian(2.0, 1.5)
        InverseGaussian(2.0, 7.0)
    ]
        test_distr(d, 10^6, test_scalar_rand = true)
    end
end