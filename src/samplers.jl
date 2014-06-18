# delegation of samplers

sampler(d::Binomial) = BinomialPolySampler(d.size, d.prob)
sampler(d::Gamma) = GammaMTSampler(d.shape, d.scale)

