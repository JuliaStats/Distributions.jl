struct ExponentialSampler <: Sampleable{Univariate,Continuous}
    scale::AbstractFloat
end

rand(rng::AbstractRNG, s::ExponentialSampler) = s.scale * randexp(rng)

struct ExponentialLogUSampler <: Sampleable{Univariate,Continuous}
    scale::AbstractFloat
end

rand(rng::AbstractRNG, s::ExponentialLogUSampler) = -s.scale * log(rand(rng))
