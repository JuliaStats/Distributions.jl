using Distributions
using Test

@testset "Dirac tests" begin
    for val in (3, 3.0, -3.5)
        d = Dirac(val)

        @test minimum(d) == val
        @test maximum(d) == val
        @test !insupport(d, prevfloat(float(val)))
        @test insupport(d, val)
        @test !insupport(d, nextfloat(float(val)))
        @test support(d) == (val,)

        @test iszero(pdf(d, prevfloat(float(val))))
        @test isone(pdf(d, val))
        @test iszero(pdf(d, nextfloat(float(val))))

        @test logpdf(d, prevfloat(float(val))) == -Inf
        @test iszero(logpdf(d, val))
        @test logpdf(d, nextfloat(float(val))) == -Inf

        @test iszero(cdf(d, prevfloat(float(val))))
        @test isone(cdf(d, val))
        @test isone(cdf(d, nextfloat(float(val))))

        @test quantile(d, 0) == val
        @test quantile(d, 0.5) == val
        @test quantile(d, 1) == val

        @test rand(d) == val

        @test mean(d) == val
        @test iszero(var(d))

        @test mode(d) == val

        @test iszero(entropy(d))

        t = rand()
        @test mgf(d, t) == exp(t * val)
        @test cf(d, t) == cis(t * val)
    end
end
