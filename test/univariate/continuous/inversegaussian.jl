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

@testset "InverseGaussian quantile" begin
    p(num_σ) = erf(num_σ/sqrt(2))

    begin
        dist = InverseGaussian{Float64}(1.187997687788096, 60.467382225458564)
        @test quantile(dist, p(4)) ≈ 1.9990956368573651
        @test quantile(dist, p(5)) ≈ 2.295607340999747
        @test quantile(dist, p(6)) ≈ 2.6249349452113484
    end

    @test quantile(InverseGaussian{Float64}(17.84806245738152, 163.707062977564), 0.9999981908772995) ≈ 69.37000274656731

    begin
        dist = InverseGaussian(1.0, 0.25)
        @test quantile(dist, 0.99) ≈ 9.90306205018232
        @test quantile(dist, 0.999) ≈ 21.253279722084798
        @test quantile(dist, 0.9999) ≈ 34.73673452136752
        @test quantile(dist, 0.99999) ≈ 49.446586395457985
        @test quantile(dist, 0.999996) ≈ 55.53114044452607
        @test quantile(dist, 0.999999) ≈ 64.92521558088777
    end
end
