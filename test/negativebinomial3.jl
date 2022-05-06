using Distributions
using Test

@testset "Check the corner case p==1" begin
    d = convert(NegativeBinomial3{Float64}, NegativeBinomial(0.5, 1.0))
    @test logpdf(d, 0) === 0.0
    @test logpdf(d, 1) === -Inf
end

@testset "Interconversion checks" begin
    d = NegativeBinomialPoissonGamma()
    nb = NegativeBinomial()
    nbl = NegativeBinomialLocation()
    nbll = NegativeBinomialLogLocation()
    nbpg = NegativeBinomialPoissonGamma()
    for (d₂, T) ∈ zip((nb, nbl, nbll, nbpg), (NegativeBinomial, NegativeBinomialLocation, NegativeBinomialLogLocation, NegativeBinomialPoissonGamma))
        # Convert to other distribution from NegativeBinomialPoissonGamma
        @test d₂ == convert(T{Float64}, d)
        # Convert from other distribution to NegativeBinomialPoissonGamma
        @test d == convert(NegativeBinomialPoissonGamma{Float64}, d₂)
    end
end

@testset "logpdf checks" begin
    d = NegativeBinomialPoissonGamma()
    nb = NegativeBinomial()
    nbl = NegativeBinomialLocation()
    nbll = NegativeBinomialLogLocation()
    nbpg = NegativeBinomialPoissonGamma()
    for d₂ ∈ (nb, nbl, nbll, nbpg), n ∈ (2, 10, 10^2, 10^5, 10^8)
        @test logpdf(d, n) ≈ logpdf(d₂, n)
    end
end
