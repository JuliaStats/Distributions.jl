using Distributions
using Test

@testset "Log of Beta-binomial distribution" begin
    d = BetaBinomial(50, 0.2, 0.6)

    for k in 1:50
        p  = pdf(d, k)
        lp = logpdf(d, k)
        @test lp ≈ log(p)
        @test insupport(d, k)
    end
    @test !insupport(d, 51)
end
