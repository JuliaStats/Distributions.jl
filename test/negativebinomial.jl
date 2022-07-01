using Distributions
using ChainRulesTestUtils
using FiniteDifferences
using Test, ForwardDiff

# Currently, most of the tests for NegativeBinomial are in the "ref" folder.
# Eventually, we might want to consolidate the tests here

mydiffp(r, p, k) = isone(p) ? r : r/p - k/(1 - p)
mydiffr(r, p, k) = log(p) - inv(k + r) - digamma(r) + digamma(r + k + 1)

@testset "NegativeBinomial" begin
    ks = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    @testset "logpdf and ForwardDiff" begin
        rs = exp10.(range(-10, stop=2, length=25))
        ps = exp10.(-10:0) .- eps() # avoid p==1 since it's not differentiable
        @testset "r=$r" for r in rs
            @testset "p=1, k=0" begin
                @test logpdf(NegativeBinomial(r, 1.0), 0) === 0.0
                fdm = backward_fdm(5, 1, max_range = r/10)
                f1(_p) = logpdf(NegativeBinomial(r, _p), 0)
                f2(_r) = logpdf(NegativeBinomial(_r, 1.0), 0)
                @test_broken ForwardDiff.derivative(f1, 1.0) ≈ mydiffp(r, 1.0, 0)
                @test_skip ForwardDiff.derivative(f2, r) ≈ mydiffr(r, 1.0, 0)
            end
            @testset "p=1, k=1" begin
                @test logpdf(NegativeBinomial(r, 1.0), 1) === -Inf
            end
            @testset "p=$p, k=$k" for p in ps, k in ks
                fdm = central_fdm(5, 1, max_range = min(r, p, 1-p)/2)
                f1(_p) = logpdf(NegativeBinomial(r, _p), k)
                f2(_r) = logpdf(NegativeBinomial(_r, p), k)
                @test ForwardDiff.derivative(f1, p) ≈ mydiffp(r, p, k)
                @test_skip ForwardDiff.derivative(f2, r) ≈ mydiffr(r, p, k)
            end
        end
    end
    @testset "rrule" begin
        rs = -log.(rand(25))
        ps = rand(25)
        @testset "r=$r" for r in rs
            @testset "p=1, k=0" begin
                dist = NegativeBinomial(r, 1.0)
                fdm = backward_fdm(5, 1, max_range = r/10)
                test_rrule(logpdf, dist, 0; fdm = fdm, check_inferred=true)
            end
            @testset "p=$p, k=$k" for p in ps, k in ks
                dist = NegativeBinomial(r, p)
                fdm = central_fdm(5, 1, max_range = min(r, p, 1-p)/2)
                test_rrule(logpdf, dist, k; fdm = fdm, check_inferred=true)
            end
        end
    end
end
