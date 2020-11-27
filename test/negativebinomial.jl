using Distributions
using Test, ForwardDiff

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
    @test logpdf(NegativeBinomial(0.5, 1.0), 0) === 0.0
    @test logpdf(NegativeBinomial(0.5, 1.0), 1) === -Inf
end
