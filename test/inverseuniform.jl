import Random
using Test
using Distributions

@testset "inverseuniform" begin
    rng = Random.MersenneTwister(0)
    a = 0.1
    b = 1
    U = Uniform(a,b)
    IU = InverseUniform(inv(b), inv(a))
    N = 10^4
    sample1 = map(inv, rand(rng,U, N))
    sample2 = [rand(rng, IU) for _ in 1:N]
    @test pvalue_kolmogorovsmirnoff(sample1, IU) > 1e-2
    @test pvalue_kolmogorovsmirnoff(sample2, IU) > 1e-2

    test_distr(InverseUniform(0.1, 1), 10000, rng=rng)
    test_distr(InverseUniform(0.1, 0.2), 10000, rng=rng)
    for _ in 1:100
        a = rand(rng,Uniform(0.1,10))
        b = rand(rng,Uniform(0.1,10))
        (a,b) = minmax(a,b)
        x = rand(rng, Uniform(a,b))
        y = inv(x)
        d_inv = InverseUniform(inv(b), inv(a))
        py = pdf(d_inv, y)
        px = pdf(Uniform(a,b), x)
        @test py <= pdf(d_inv, mode(d_inv))
        @test px * (y^-2) ≈ py
    end
end
