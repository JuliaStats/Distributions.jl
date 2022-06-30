using Distributions
using ChainRulesTestUtils
using FiniteDifferences
using Test, ForwardDiff

# Currently, most of the tests for NegativeBinomial are in the "ref" folder.
# Eventually, we might want to consolidate the tests here

@testset "NegativeBinomial" begin
    @testset "ForwardDiff and rrule" begin
        rs = -log.(rand(25))
        ps = rand(25)
        ks = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
        @testset "r=$r" for r in rs
            @testset "p=1, k=0" begin
                dist = NegativeBinomial(r, 1.0)
                fdm = backward_fdm(5, 1, max_range = r/10)
                f1(_p) = logpdf(NegativeBinomial(r, _p), 0)
                f2(_r) = logpdf(NegativeBinomial(_r, 1.0), 0)
                @test ForwardDiff.derivative(f1, 1.0) ≈ fdm(f1, 1.0)
                @test ForwardDiff.derivative(f2, r) ≈ fdm(f2, r)
                test_rrule(logpdf, dist, 0; fdm = fdm, check_inferred=true)
            end
            @testset "p=$p, k=$k" for p in ps, k in ks
                dist = NegativeBinomial(r, p)
                fdm = central_fdm(5, 1, max_range = min(r, p)/2)
                f1(_p) = logpdf(NegativeBinomial(r, _p), k)
                f2(_r) = logpdf(NegativeBinomial(_r, p), k)
                @test ForwardDiff.derivative(f1, p) ≈ fdm(f1, p)
                @test ForwardDiff.derivative(f2, r) ≈ fdm(f2, r)
                test_rrule(logpdf, dist, k; fdm = fdm, check_inferred=true)
            end
        end
    end
end

@testset "Check the corner case p==1" begin
    @test logpdf(NegativeBinomial(0.5, 1.0), 0) === 0.0
    @test logpdf(NegativeBinomial(0.5, 1.0), 1) === -Inf
end
