using Distributions
using Test, ForwardDiff
using ChainRulesTestUtils
using FiniteDifferences

# Currently, most of the tests for NegativeBinomial are in the "ref" folder.
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
    @test all(iszero, rand(NegativeBinomial(rand(), 1.0), 10))
end

@testset "rrule: logpdf of NegativeBinomial" begin
    r = randexp()

    # Test with values in and outside of support
    p = rand()
    dist = NegativeBinomial(r, p)
    fdm = central_fdm(5, 1; max_range=min(r, p, 1-p)/2) # avoids numerical issues with finite differencing
    for k in (0, 10, 42, -1, -5, -13)
        # Test both integers and floating point numbers.
        # For floating point numbers we have to tell FiniteDifferences explicitly that the
        # argument is non-differentiable. Otherwise it will compute `NaN` as derivative.
        test_rrule(logpdf, dist, k; fdm=fdm, nans=true)
        test_rrule(logpdf, dist, float(k) ⊢ ChainRulesTestUtils.NoTangent(); fdm=fdm, nans=true)
    end

    # Test edge case `p = 1` and `k = 0`
    dist = NegativeBinomial(r, 1)
    fdm = backward_fdm(5, 1; max_range = r/10)
    test_rrule(logpdf, dist, 0; fdm=fdm)
    test_rrule(logpdf, dist, 0.0 ⊢ ChainRulesTestUtils.NoTangent(); fdm=fdm)
end
