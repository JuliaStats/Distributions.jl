using Test
using Distributions
using ForwardDiff

@testset "Arcsine" begin
    @test_throws ArgumentError Arcsine(5, 3)
    d = Arcsine(3, 5)
    d2 = Arcsine(3.5f0, 5)
    @test partype(d) == Float64
    @test partype(d2) == Float32
    @test d == deepcopy(d)

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
