using Test, Distributions, SpecialFunctions

@testset "pdf L2 norm" begin
    # Test error on a non implemented norm.
    @test_throws ArgumentError pdfL2norm(Gumbel())

    @testset "Beta" begin
        @test pdfL2norm(Beta(1, 1)) ≈ 1
        @test pdfL2norm(Beta(2, 2)) ≈ 6 / 5
        @test pdfL2norm(Beta(0.25, 1)) ≈ Inf
        @test pdfL2norm(Beta(1, 0.25)) ≈ Inf
    end

    @testset "Cauchy" begin
        @test pdfL2norm(Cauchy(0, 1)) ≈ 1 / (2 * π)
        @test pdfL2norm(Cauchy(0, 2)) ≈ 1 / (4 * π)
        # The norm doesn't depend on the mean
        @test pdfL2norm(Cauchy(100, 1)) == pdfL2norm(Cauchy(-100, 1)) == pdfL2norm(Cauchy(0, 1))
    end

    @testset "Chi" begin
        @test pdfL2norm(Chi(2)) ≈ gamma(3 / 2) / 2
        @test pdfL2norm(Chi(0.25)) ≈ Inf
    end

    @testset "Chisq" begin
        @test pdfL2norm(Chisq(2)) ≈ 1 / 4
        @test pdfL2norm(Chisq(1)) ≈ Inf
    end

    @testset "Exponential" begin
        @test pdfL2norm(Exponential(1)) ≈ 1 / 2
        @test pdfL2norm(Exponential(2)) ≈ 1 / 4
    end

    @testset "Gamma" begin
        @test pdfL2norm(Gamma(1, 1)) ≈ 1 / 2
        @test pdfL2norm(Gamma(1, 2)) ≈ 1 / 4
        @test pdfL2norm(Gamma(2, 2)) ≈ 1 / 8
        @test pdfL2norm(Gamma(1, 0.25)) ≈ 2
        @test pdfL2norm(Gamma(0.5, 1)) ≈ Inf
    end

    @testset "Logistic" begin
        @test pdfL2norm(Logistic(0, 1)) ≈ 1 / 6
        @test pdfL2norm(Logistic(0, 2)) ≈ 1 / 12
        # The norm doesn't depend on the mean
        @test pdfL2norm(Logistic(100, 1)) == pdfL2norm(Logistic(-100, 1)) == pdfL2norm(Logistic(0, 1))
    end

    @testset "Normal" begin
        @test pdfL2norm(Normal(0, 1)) ≈ 1 / (2 * sqrt(π))
        @test pdfL2norm(Normal(0, 2)) ≈ 1 / (4 * sqrt(π))
        @test pdfL2norm(Normal(1, 0)) ≈ Inf
        # The norm doesn't depend on the mean
        @test pdfL2norm(Normal(100, 1)) == pdfL2norm(Normal(-100, 1)) == pdfL2norm(Normal(0, 1))
    end

    @testset "Uniform" begin
        @test pdfL2norm(Uniform(-1, 1)) ≈ 1 / 2
        @test pdfL2norm(Uniform(1, 2)) ≈ 1
    end
end
