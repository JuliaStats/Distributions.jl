using Distributions
using Test, ForwardDiff
using StatsFuns

# Currently, most of the tests for NegativeBinomail are in the "ref" folder.
# Eventually, we might want to consolidate the tests here

mydiffp(r, p, k) = r/p - k/(1 - p)

@testset "NegativeBinomial r=$r, p=$p, k=$k" for
    p in exp10.(-10:0) .- eps(), # avoid p==1 since it's not differentiable
        r in exp10.(range(-10, stop=2, length=25)),
            k in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

    @test ForwardDiff.derivative(_p -> logpdf(NegativeBinomial(r, _p), k), p) ≈ mydiffp(r, p, k) rtol=1e-12 atol=1e-12
end

@testset "Check the corner case p==1" begin
    for r in randexp(10)
        d = NegativeBinomial(r, 1.0)
        @test @inferred(logpdf(d, 0)) === 0.0
        @test @inferred(logpdf(d, -1)) === -Inf
        @test @inferred(logpdf(d, 1)) === -Inf
        @test all(iszero, rand(d, 10))
    end
end

@testset "Check the corner case k==0" begin
	for r in randexp(5), p in rand(5)
        @test @inferred(logpdf(NegativeBinomial(r, p), 0)) === xlogy(r, p)
    end
end
