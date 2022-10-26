using Distributions
using Test, ForwardDiff
using ChainRulesTestUtils
using FiniteDifferences
using StatsFuns

# Currently, most of the tests for NegativeBinomial are in the "ref" folder.
# Eventually, we might want to consolidate the tests here

test_cgf(NegativeBinomial(10,0.5), (-1f0, -200.0,-1e6))
test_cgf(NegativeBinomial(3,0.1),  (-1f0, -200.0,-1e6))

mydiffp(r, p, k) = iszero(k) ? r/p : r/p - k/(1 - p)
mydiffr(r, p, k) = iszero(k) ? log(p) : log(p) - inv(k + r) - digamma(r) + digamma(r + k + 1)

@testset "issue #1603" begin
    d = NegativeBinomial(4, 0.2)
    fdm = central_fdm(5, 1)
    @test fdm(Base.Fix1(mgf, d), 0) ≈ mean(d)
    d = NegativeBinomial(1, 0.2)
    @test fdm(Base.Fix1(mgf, d), 0) ≈ mean(d)
    @test fdm(Base.Fix1(cf, d), 0) ≈ mean(d) * im

    fdm2 = central_fdm(5, 2)
    m2 = var(d) + mean(d)^2
    @test fdm2(Base.Fix1(mgf, d), 0) ≈ m2
    @test fdm2(Base.Fix1(cf, d), 0) ≈ -m2
end

@testset "NegativeBinomial r=$r, p=$p, k=$k" for
    p in exp10.(-10:0) .- eps(), # avoid p==1 since it's not differentiable
        r in exp10.(range(-10, stop=2, length=25)),
            k in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

    @test ForwardDiff.derivative(_p -> logpdf(NegativeBinomial(r, _p), k), p) ≈ mydiffp(r, p, k) rtol=1e-12 atol=1e-12
    @test ForwardDiff.derivative(_r -> logpdf(NegativeBinomial(_r, p), k), r) ≈ mydiffr(r, p, k) rtol=1e-12 atol=1e-12
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

@testset "issue #1582" begin
    dp = mydiffp(1.0, 1.0, 0.0)
    @test ForwardDiff.derivative(p -> logpdf(NegativeBinomial(1.0, p), 0.0), 1.0) == dp == 1.0

    dr = mydiffr(1.0, 1.0, 0.0)
    @test ForwardDiff.derivative(r -> logpdf(NegativeBinomial(r, 1.0), 0.0), 1.0) == dr == 0.0
end
