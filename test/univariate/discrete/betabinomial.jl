using Distributions
using Test

@testset "Log of Beta-binomial distribution" begin
    d = BetaBinomial(50, 0.2,0.6)

    for k in Base.OneTo(50)
        p  = pdf(d, k)
        lp = logpdf(d, k)
        @test_approx_eq lp log(p)
        @test insupport(d, k)
    end
    @test !insupport(d, 51)
end
