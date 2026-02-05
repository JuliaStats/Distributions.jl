using Distributions
using Test

@testset "Hypergeometric unimodal or bimodal" begin
    # bimodal  if ns == nf and n is odd, else unimodal
    for ns ∈ 1:10, nf ∈ 1:10, n ∈ 0:ns+nf
        @test length(modes(Hypergeometric(ns, nf, n))) == (ns == nf && isodd(n) ? 2 : 1)
    end
end