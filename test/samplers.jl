# Testing of samplers

using Distributions
using Base.Test

import Distributions: 
    CategoricalDirectSampler, 
    AliasTable,
    BinomialGeomSampler, 
    BinomialTPESampler, 
    BinomialPolySampler, 
    BinomialAliasSampler, 
    PoissonADSampler, 
    PoissonCountSampler,
    ExponentialSampler,
    GammaGDSampler,
    GammaGSSampler,
    GammaMTSampler,
    GammaIPSampler

n_tsamples = 10^6

## Categorical samplers

@test_throws ErrorException CategoricalDirectSampler(Float64[])
@test_throws ErrorException AliasTable(Float64[])

for S in [CategoricalDirectSampler, AliasTable]
    println("    testing $S")
    for p in Any[[1.0], [0.3, 0.7], [0.2, 0.3, 0.4, 0.1]]
        test_samples(S(p), Categorical(p), n_tsamples)
    end
end


## Binomial samplers

binomparams = [(0, 0.4), (0, 0.6), (5, 0.0), (5, 1.0), 
               (1, 0.2), (1, 0.8), (3, 0.4), (4, 0.6), 
               (40, 0.5), (100, 0.4), (300, 0.6)]

for (S, paramlst) in [
    (BinomialGeomSampler, [(0, 0.4), (0, 0.6), (5, 0.0), (5, 1.0), (1, 0.2), (1, 0.8), (3, 0.4), (4, 0.6)]), 
    (BinomialTPESampler, [(40, 0.5), (100, 0.4), (300, 0.6)]),
    (BinomialPolySampler, binomparams),
    (BinomialAliasSampler, binomparams) ]

    println("    testing $S")
    for pa in paramlst
        n, p = pa
        test_samples(S(n, p), Binomial(n, p), n_tsamples)
    end
end


## Poisson samplers

for (S, paramlst) in [
    (PoissonCountSampler, [0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0]), 
    (PoissonADSampler, [5.0, 10.0, 15.0, 20.0, 30.0])]

    println("    testing $S")
    for μ in paramlst
        test_samples(S(μ), Poisson(μ), n_tsamples)
    end
end


## Exponential samplers

for S in [ExponentialSampler]
    println("    testing $S")
    for scale in [1.0, 2.0, 3.0]
        test_samples(S(scale), Exponential(scale), n_tsamples)
    end
end


## Gamma samplers
# shape >= 1
for S in [GammaGDSampler, GammaMTSampler]
    println("    testing $S")
    for d in [Gamma(1.0, 1.0), Gamma(2.0, 1.0), Gamma(3.0, 1.0),
               Gamma(1.0, 2.0), Gamma(3.0, 2.0), Gamma(100.0, 2.0)]
        test_samples(S(d), d, n_tsamples)
    end
end

# shape < 1
for S in [GammaGSSampler, GammaIPSampler]
    println("    testing $S")
    for d in [Gamma(0.1,1.0),Gamma(0.9,1.0)]
        test_samples(S(d), d, n_tsamples)
    end
end
