using Distributions, Random, Test

@testset "truncated exponential" begin
    d = Exponential(1.5)
    l = 1.2
    r = 2.7
    @test mean(d) ≈ mean(truncated(d; lower=-3.0)) # non-binding truncation
    @test mean(d) ≈ mean(truncated(d; lower=-3.0, upper=Inf))
    @test mean(truncated(d; lower=l)) ≈ mean(d) + l
    @test mean(truncated(d; lower=l, upper=Inf)) ≈ mean(d) + l
    # test values below calculated using symbolic integration in Maxima
    @test mean(truncated(d, 0, r)) ≈ 0.9653092084094841
    @test mean(truncated(d, l, r)) ≈ 1.82703493969601

    # all the fun corner cases and numerical quirks
    @test mean(truncated(Exponential(1.0); upper=0)) == 0                   # degenerate
    @test mean(truncated(Exponential(1.0); lower=-Inf, upper=0)) == 0
    @test mean(truncated(Exponential(1.0); upper=0+eps())) ≈ 0 atol = eps() # near-degenerate
    @test mean(truncated(Exponential(1.0); lower=-Inf, upper=0+eps())) ≈ 0 atol=eps()
    @test mean(truncated(Exponential(1.0), 1.0, 1.0+eps())) ≈ 1.0 # near-degenerate
    @test mean(truncated(Exponential(1e308), 1.0, 1.0+eps())) ≈ 1.0 # near-degenerate
end
