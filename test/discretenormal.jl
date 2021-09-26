using Distributions

@testset "DiscreteNormal" begin
    dn = DiscreteNormal(0, 8/√(2π))

    rng = MersenneTwister(123)
    # Try the various available samplers, just to make sure they don't error
    let samp = Distributions.DiscreteNormalGeneric(dn, Distributions.AliasTable)
        rand(rng, samp, 10000)
    end
    let samp = Distributions.DiscreteNormalGeneric(dn, Distributions.InversionCDT)
        rand(rng, samp, 10000)
    end
    let samp = Distributions.Karney.DiscreteNormalKarney(dn)
        rand(rng, samp, 10000)
    end
end
