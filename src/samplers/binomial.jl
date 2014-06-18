sampler(d::Binomial) = BinomialPolySampler(d.size, d.prob)

rand(d::Binomial) = rand(sampler(d))
rand!(d::Binomial,a::Array) = rand!(sampler(d),a)

