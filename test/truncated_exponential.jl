using Distributions, Random, Test

@testset "truncated exponential" begin
    d = Exponential(1.5)
    l = 1.2
    r = 2.7
    @test mean(d) ≈ mean(Truncated(d, -3.0, Inf)) # non-binding truncation
    @test mean(Truncated(d, l, Inf)) ≈ mean(d) + l
    # test values below calculated using symbolic integration in Maxima
    @test mean(Truncated(d, 0, r)) ≈ 0.9653092084094841
    @test mean(Truncated(d, l, r)) ≈ 1.82703493969601
end
