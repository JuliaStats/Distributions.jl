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
    GammaMTSampler

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

for S in [GammaMTSampler]
    for pa in [(1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (0.5, 1.0), 
               (1.0, 2.0), (3.0, 2.0), (0.5, 2.0)]
        test_samples(S(pa...), Gamma(pa...), n_tsamples)
    end
end

