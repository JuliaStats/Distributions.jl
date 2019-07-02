@testset "discrete univariate" begin

    @testset "Bernoulli" begin
        d1 = Bernoulli(0.1)
        d2 = convolve(d1, d1)

        @test isa(d2, Binomial)
        @test d2.n == 2
        @test d2.p == 0.1

        # cannot convolve a Binomial with a Bernoulli
        @test_throws MethodError convolve(d1, d2)

        # only works if p1 == p2
        d3 = Bernoulli(0.2)
        @test_throws ArgumentError convolve(d1, d3)

    end

    @testset "Binomial" begin
        d1 = Binomial(2, 0.1)
        d2 = Binomial(5, 0.1)
        d3 = convolve(d1, d2)

        @test isa(d3, Binomial)
        @test d3.n == 7
        @test d3.p == 0.1

        # only works if p1 == p2
        d4 = Binomial(2, 0.2)
        @test_throws ArgumentError convolve(d1, d4)

    end

    @testset "NegativeBinomial" begin
        d1 = NegativeBinomial(4, 0.1)
        d2 = NegativeBinomial(1, 0.1)
        d3 = convolve(d1, d2)

        isa(d3, NegativeBinomial)
        @test d3.r == 5
        @test d3.p == 0.1

        d4 = NegativeBinomial(1, 0.2)
        @test_throws ArgumentError convolve(d1, d4)
    end


    @testset "Geometric" begin
        d1 = Geometric(0.2)
        d2 = convolve(d1, d1)

        @test isa(d2, NegativeBinomial)
        @test d2.p == 0.2

        # cannot convolve a Geometric with a NegativeBinomial
        @test_throws MethodError convolve(d1, d2)

        # only works if p1 == p2
        d3 = Geometric(0.5)
        @test_throws ArgumentError convolve(d1, d3)
    end

    @testset "Poisson" begin
        d1 = Poisson(0.1)
        d2 = Poisson(0.4)
        d3 = convolve(d1, d2)

        @test isa(d3, Poisson)
        @test d3.λ == 0.5
    end

end

@testset "continuous univariate" begin

    @testset "Gaussian" begin
        d1 = Normal(0.1, 0.2)
        d2 = Normal(0.25, 1.7)
        d3 = convolve(d1, d2)

        @test isa(d3, Normal)
        @test d3.μ == 0.35
        @test d3.σ == hypot(0.2, 1.7)
    end

    @testset "Cauchy" begin
        d1 = Cauchy(0.2, 0.7)
        d2 = Cauchy(1.9, 0.8)
        d3 = convolve(d1, d2)

        @test isa(d3, Cauchy)
        @test d3.μ == 2.1
        @test d3.σ == 1.5
    end

    @testset "Chisq" begin
        d1 = Chisq(0.1)
        d2 = Chisq(0.3)
        d3 = convolve(d1, d2)

        @test isa(d3, Chisq)
        @test d3.ν == 0.4
    end

    @testset "Exponential" begin
        d1 = Exponential(0.7)
        d2 = convolve(d1, d1)

        @test isa(d2, Gamma)
        @test d2.α == 2
        @test d2.θ == 0.7

        # cannot convolve an Exponential with a Gamma
        @test_throws MethodError convolve(d1, d2)

        # only works if θ1 == θ2
        d3 = Exponential(0.2)
        @test_throws ArgumentError convolve(d1, d3)
    end

    @testset "Gamma" begin
        d1 = Gamma(0.1, 1.7)
        d2 = Gamma(0.5, 1.7)
        d3 = convolve(d1, d2)

        @test isa(d3, Gamma)
        @test d3.α == 0.6
        @test d3.θ == 1.7

        # only works if θ1 == θ4
        d4 = Gamma(1.2, 0.4)
        @test_throws ArgumentError convolve(d1, d4)
    end

end

@testset "continuous multivariate" begin
end
