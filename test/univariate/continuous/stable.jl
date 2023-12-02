using Test, Distributions

@testset "Stable" begin
    
    @testset "input" begin
        @test_throws DomainError Stable(-3.)
        @test_throws DomainError Stable(3.)
        @test_throws DomainError Stable(1., -2.)
        @test_throws DomainError Stable(1., 2.)
        @test_throws DomainError Stable(2, 0.1)
        @test_throws DomainError Stable{Float64}(1.5, 0, -2, 0)

        @test Stable(1, 1, 1, 1) isa Stable{Float64}
        @test Stable(1.7) == Stable(1.7, 0, 1, 0)
        @test Stable(0.2, -0.5) == Stable(0.2, -0.5, 1, 0)
    end

    @testset "conversions" begin
        @test convert(Stable{Float64}, 1, 1, 1, 1) isa Stable{Float64}
        @test convert(Stable{BigFloat}, Stable(1., 1, 1, 1)) isa Stable{BigFloat}
    end

    @testset "parameters" begin
        @test shape(Stable(1.2, 1.)) == (1.2, 1.)
        @test location(Stable(1., 0., 1., 3.)) == 3.
        @test scale(Stable(1.5, 1., 2., 0.)) == 2.
        @test params(Stable(0.3, 0.4, 0.5, 0.6)) == (0.3, 0.4, 0.5, 0.6)
        @test partype(Stable{Int}(1, 1, 1, 1)) === Int
    end

    @testset "statistics and support" begin
        @test mean(Stable(2., 0., 3., -5.)) == -5.
        @test mean(Stable(1.1, 0., 3., 42.)) == 42.
        @test isnan(mean(Stable(1, 1)))
        @test var(Stable(2., 0., 3., 0.)) == 18.
        @test isinf(var(Stable(1.9, 0., 3., 0.)))
        @test iszero(skewness(Stable(2.0, 0., 2., 1.)))
        @test isnan(skewness(Stable(1.5, 0., 2., 1.)))
        @test iszero(kurtosis(Stable(2.0, 0., 2., 1.)))
        @test isnan(kurtosis(Stable(1.5, 0., 2., 1.)))

        @test support(Stable(1.1)) == RealInterval(-Inf, Inf)
        @test support(Stable(1., 1.)) == RealInterval(-Inf, Inf)
        @test support(Stable(0.9, 1.)) == RealInterval(0., Inf)
        @test support(Stable(0.9, -1., 1., 2.)) == RealInterval(-Inf, 2.)
        @test support(Stable(0.9, 0.9)) == RealInterval(-Inf, Inf)
        @test support(Stable(0.9, -0.9, 1., 2.)) == RealInterval(-Inf, Inf)
    end

    # we cannot use test_affine_transformations at α == 1
    @testset "affine transformations" begin
        @test 2Stable(0.7, -1, 1, 1) + 1 == Stable(0.7, -1, 2, 3)
        @test -2Stable(1.5, -1, 1, 1) + 1 == Stable(1.5, 1, 2, -1)
        @test -3Stable(1, 1, 2, 1) + 2 ≈ Stable(1, -1, 6, -3 + 2 + 2/pi*6*log(3))
    end

    @testset "pdf, cdf and mgf in special cases" begin
        xs = LinRange(-5, 5, 20)
        @test pdf.(Stable(2., 0., 2., 1.), xs) ≈ pdf.(Normal(1., 2√2), xs)
        @test cdf.(Stable(2., 0., 2., 1.), xs) ≈ cdf.(Normal(1., 2√2), xs)
        @test pdf.(Stable(1., 0., 3., -1.), xs) == pdf.(Cauchy(-1., 3.), xs)
        @test cdf.(Stable(1., 0., 3., -1.), xs) == cdf.(Cauchy(-1., 3.), xs)
        @test pdf.(Stable(0.5, 1., 2., 1.), xs) == pdf.(Levy(1., 2.), xs)
        @test cdf.(Stable(0.5, 1., 2., 1.), xs) == cdf.(Levy(1., 2.), xs)
        @test pdf.(Stable(0.5, -1., 2., 3.), xs) == pdf.(Levy(-3., 2.), -xs)
        @test cdf.(Stable(0.5, -1., 2., 3.), xs) ≈ 1 .- cdf.(Levy(-3., 2.), -xs)
        @test mgf.(Stable(2., 0., 2., -2.), xs) == mgf.(Normal(-2., √2*2), xs)
    end

    # test values taken from Mathematica 13.2
    @testset "pdf, cdf and cf in general case" begin
        @test cf(Stable(0.5), -2) ≈ exp(-√2)
        @test cf(Stable(1.5, -0.5, 0.5, 0.5), 4.2) ≈ exp(im*2.1 - (2.1)^1.5*(1 + im*0.5*tan(0.75*π)))
        @test cf(Stable(1., -0.75, 1., 0.), -4.2) ≈ exp( -(4.2)*(1 + im*1.5/π*log(4.2)))
        @test isinf(mgf(Stable(1.9), 42.))

        # checking proper shape, Stable(α, β)
        @test pdf(Stable(1.5), 0.) ≈ 0.287353 atol = 1e-6
        @test pdf(Stable(1.1), 0.) ≈ 0.307141 atol = 1e-6
        @test pdf(Stable(0.4), 0.) ≈ 1.057855 atol = 1e-6
        @test pdf(Stable(0.2), 0.) ≈ 38.197186 atol = 1e-6
        @test pdf(Stable(1.9, 1.), 3.) ≈ 0.0300991 atol = 1e-6
        @test pdf(Stable(1.2, 1.), -1.) ≈ 0.0973176 atol = 1e-6
        @test pdf(Stable(1., -0.9), -1.) ≈ 0.162803 atol = 1e-6
        @test pdf(Stable(1., -0.5), -10.) ≈ 0.005098 atol = 1e-6
        @test pdf(Stable(0.5, 0.5), -10.) ≈ 0.002181 atol = 1e-6
        @test pdf(Stable(0.7, 1.), -0.01) ≈ 0.000000 atol = 1e-6

        @test cdf(Stable(1.5), 0.) ≈ 0.500000 atol = 1e-6
        @test cdf(Stable(0.2), 0.) ≈ 0.500000 atol = 1e-6
        @test cdf(Stable(1.9, 1), 3.) ≈ 0.970814 atol = 1e-6
        @test cdf(Stable(1.2, 1), -1.) ≈ 0.758715 atol = 1e-6
        @test cdf(Stable(1, 0.5), -1.) ≈ 0.165443 atol = 1e-6
        @test cdf(Stable(1., -0.9), -1.) ≈ 0.405070 atol = 1e-6
        @test cdf(Stable(1., -0.5), -10.) ≈ 0.050327 atol = 1e-6
        @test cdf(Stable(0.5, 0.5), -10.) ≈ 0.052669 atol = 1e-6
        @test cdf(Stable(0.7, 1.), -0.01) ≈ 0.000000 atol = 1e-6

        # checking affine transformations, Stable(α, β, σ, μ)
        (σ, μ) = 3., -2.
        @test pdf(Stable(1.1, 1., σ, μ), 2.5) ≈ 1/σ * pdf(Stable(1.1, 1), (2.5 - μ)/σ)
        @test cdf(Stable(0.8, 0.1, σ, μ), 2.5) ≈  cdf(Stable(0.8, 0.1), (2.5 - μ)/σ)
        @test pdf(Stable(1., -0.5, σ, μ), 2.5) ≈ 1/σ * pdf(Stable(1., -0.5), (2.5 - μ)/σ + 1/π*log(σ))
        @test cdf(Stable(1., -0.5, σ, μ), 2.5) ≈  cdf(Stable(1., -0.5), (2.5 - μ)/σ + 1/π*log(σ))
        @test cdf(Stable(1.4, 0.25, 2., -1.), 0.0) ≈ 0.697587 atol = 1e-6
        @test cdf(Stable(1, 0.25, 2., -1.), 0.0) ≈ 0.581518 atol = 1e-6
        @test pdf(Stable(1, 0.25, 2., -1.), 0.0) ≈ 0.128031 atol = 1e-6
    end

end