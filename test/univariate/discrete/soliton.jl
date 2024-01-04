using Distributions

@testset "Soliton" begin
    K, M, δ, atol = 100, 60, 0.2, 0
    Ω = Soliton(K, M, δ, atol)
    @test pdf(Ω, M) > pdf(Ω, M-1)
    @test pdf(Ω, M) > pdf(Ω, M+1)
    @test cumsum(Base.Fix1(pdf, Ω).(1:K)) ≈ Base.Fix1(cdf, Ω).(1:K)
    @test cdf(Ω, 0) ≈ 0
    @test cdf(Ω, K) ≈ 1
    @test mean(Ω) ≈ 7.448826535558562
    @test var(Ω) ≈ 178.91717915136957
    @test minimum(Ω) == 1
    @test maximum(Ω) == K
    @test quantile(Ω, 0) == 1
    @test quantile(Ω, 1) == K
    @test insupport(Ω, 1) && insupport(Ω, 2) && insupport(Ω, K)
    @test !insupport(Ω, 0) && !insupport(Ω, 2.1) && !insupport(Ω, K + 1)

    K, M, δ, atol = 100, 60, 0.2, 1e-3
    Ω = Soliton(K, M, δ, atol)
    ds = [d for d in 1:K if pdf(Ω, d) > 0]
    @test all(Base.Fix1(pdf, Ω).(ds) .> atol)
    @test cdf(Ω, 0) ≈ 0
    @test cdf(Ω, K) ≈ 1
    @test minimum(Ω) == 1
    @test maximum(Ω) == K
    @test quantile(Ω, 0) == 1
    @test quantile(Ω, 1) == M
end
