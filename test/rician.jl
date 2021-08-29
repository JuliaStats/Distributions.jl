@testset "Rician" begin

    d1 = Rician(0.0, 10.0)
    @test d1 isa Rician{Float64}
    @test params(d1) == (0.0, 10.0)
    @test shape(d1) == 0.0
    @test scale(d1) == 200.0
    @test partype(d1) === Float64
    @test eltype(d1) === Float64
    @test rand(d1) isa Float64

    d2 = Rayleigh(10.0)
    @test mean(d1) ≈ mean(d2)
    @test var(d1) ≈ var(d2)
    @test mode(d1) ≈ mode(d2)
    @test median(d1) ≈ median(d2)
    @test quantile.(d1, [0.25, 0.45, 0.60, 0.80, 0.90]) ≈ quantile.(d2, [0.25, 0.45, 0.60, 0.80, 0.90])
    @test pdf.(d1, 0.0:0.1:1.0) ≈ pdf.(d2, 0.0:0.1:1.0)
    @test cdf.(d1, 0.0:0.1:1.0) ≈ cdf.(d2, 0.0:0.1:1.0)

    x = rand(Rician(5.0, 5.0), 100000)
    d1 = fit(Rician, x)
    @test d1 isa Rician{Float64}
    @test params(d1)[1] ≈ 5.0 atol=1e-1
    @test params(d1)[2] ≈ 5.0 atol=1e-1

    d1 = Rician(10.0f0, 10.0f0)
    @test d1 isa Rician{Float32}
    @test params(d1) == (10.0f0, 10.0f0)
    @test shape(d1) == 0.5f0
    @test scale(d1) == 300.0f0
    @test partype(d1) === Float32
    @test eltype(d1) === Float64
    @test rand(d1) isa Float64

    d1 = Rician()
    @test d1 isa Rician{Float64}
    @test params(d1) == (0.0, 1.0)

    d1 = Rician(K=0.5, Ω=300.0)
    @test d1 isa Rician{Float64}
    @test params(d1)[1] ≈ 10.0
    @test params(d1)[2] ≈ 10.0
    @test shape(d1) ≈ 0.5
    @test scale(d1) ≈ 300.0

end
