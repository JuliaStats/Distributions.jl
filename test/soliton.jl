using Distributions

K, M, δ, atol = 100, 60, 0.2, 0
Ω = Soliton(K, M, δ, atol)
@test pdf(Ω, M) > pdf(Ω, M-1)
@test pdf(Ω, M) > pdf(Ω, M+1)
@test cumsum(pdf.(Ω, 1:K)) ≈ cdf.(Ω, 1:K)
@test cdf(Ω, 0) ≈ 0
@test cdf(Ω, K) ≈ 1
@test mean(Ω) ≈ 7.448826535558562

K, M, δ, atol = 100, 60, 0.2, 1e-3
Ω = Soliton(K, M, δ, atol)
ds = [d for d in 1:K if pdf(Ω, d) > 0]
@test all(pdf.(Ω, ds) .> atol)
@test cdf(Ω, 0) ≈ 0
@test cdf(Ω, K) ≈ 1
