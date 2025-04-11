@testset "InverseGaussian cdf outside of [0, 1] (#1873)" begin
    for d in [
        InverseGaussian(1.65, 590),
        InverseGaussian(0.5, 1000)
    ]
        for x in [0.02, 1.0, 20.0, 300.0]
            p = cdf(d, x)
            @test 0.0 <= p <= 1.0
            @test p ≈ exp(logcdf(d, x))

            q = ccdf(d, x)
            @test 0.0 <= q <= 1.0
            @test q ≈ exp(logccdf(d, x))

            @test (p + q) ≈ 1
        end
    end
end