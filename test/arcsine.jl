using Test
using Distributions
using ForwardDiff

@testset "Arcsine" begin
    @testset "check_args" begin
        # with `check_args=true`
        @test_throws ArgumentError Arcsine(5, 3)
        @test_throws ArgumentError Arcsine(-3)
        @test_throws ArgumentError Arcsine(; a=5, b=3)
        @test_throws ArgumentError Arcsine(; a=2)
        @test_throws ArgumentError Arcsine(; b=-3)

        # with `check_args=false`
        Arcsine(5, 3; check_args=false)
        Arcsine(-3; check_args=false)
        Arcsine(; a=5, b=3, check_args=false)
        Arcsine(; b=-3, check_args=false)
        Arcsine(; a=2, check_args=false)
    end

    @testset "default arguments" begin
        @test Arcsine(3) == Arcsine(0, 3)
        @test Arcsine(; b=3) == Arcsine(0, 3)
        @test Arcsine(; a=0.5) == Arcsine(0.5, 1)
    end

    d = Arcsine(3, 5)
    @test partype(d) == Float64
    @test d == deepcopy(d)
    @test d == Arcsine(; a=3, b=5)

    d2 = Arcsine(3.5f0, 5)
    @test partype(d2) == Float32
    @test d2 == Arcsine(; a=3.5f0, b=5)

    @test logpdf(d, 4.0) ≈ log(pdf(d, 4.0))
    # out of support
    @test isinf(logpdf(d, 2.0))
    @test isinf(logpdf(d, 6.0))
    # on support limits
    @test isinf(logpdf(d, 3.0))
    @test isinf(logpdf(d, 5.0))

    # derivative
    for v in 3.01:0.1:4.99
        fgrad = ForwardDiff.derivative(x -> logpdf(d, x), v)
        glog = gradlogpdf(d, v)
        @test fgrad ≈ glog
    end
end
