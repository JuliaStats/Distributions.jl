using Distributions
using Test

@testset "Check the corner case p==1" begin
    d = convert(NegativeBinomial2{Float64}, NegativeBinomial(0.5, 1.0))
    @test logpdf(d, 0) === 0.0
    @test logpdf(d, 1) === -Inf
end

@testset "Interconversion checks" begin
    d = NegativeBinomial2()
    nb = NegativeBinomial()
    nb2 = NegativeBinomial2()
    nb2l = NegativeBinomial2Log()
    nb3 = NegativeBinomial3()
    for (d₂, T) ∈ zip((nb, nb2, nb2l, nb3), (NegativeBinomial, NegativeBinomial2, NegativeBinomial2Log, NegativeBinomial3))
        # Convert to other distribution from NegativeBinomial2
        @test d₂ == convert(T{Float64}, d)
        # Convert from other distribution to NegativeBinomial2
        @test d == convert(NegativeBinomial2{Float64}, d₂)
    end
end

@testset "logpdf checks" begin
    d = NegativeBinomial2()
    nb = NegativeBinomial()
    nb2 = NegativeBinomial2()
    nb2l = NegativeBinomial2Log()
    nb3 = NegativeBinomial3()
    for d₂ ∈ (nb, nb2, nb2l, nb3), n ∈ (2, 10, 10^2, 10^5, 10^8)
        @test logpdf(d, n) ≈ logpdf(d₂, n)
    end
end
