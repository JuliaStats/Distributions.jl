@auto_hash_equals struct ExponentialSampler <: Sampleable{Univariate,Continuous}
    scale::Float64
end

rand(rng::AbstractRNG, s::ExponentialSampler) = s.scale * randexp(rng)

@auto_hash_equals struct ExponentialLogUSampler <: Sampleable{Univariate,Continuous}
    scale::Float64
end

rand(rng::AbstractRNG, s::ExponentialLogUSampler) = -s.scale * log(rand(rng))
