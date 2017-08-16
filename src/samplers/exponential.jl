struct ExponentialSampler <: Sampleable{Univariate,Continuous}
    scale::Float64
end

rand(s::ExponentialSampler) = rand(GLOBAL_RNG, s)
rand(rng::AbstractRNG, s::ExponentialSampler) = s.scale * randexp(rng)

struct ExponentialLogUSampler <: Sampleable{Univariate,Continuous}
    scale::Float64
end

rand(s::ExponentialLogUSampler) = rand(GLOBAL_RNG, s)
rand(rng::AbstractRNG, s::ExponentialLogUSampler) = -s.scale * log(rand(rng))
