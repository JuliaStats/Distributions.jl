struct ExponentialSampler <: Sampleable{Univariate,Continuous}
    scale::Float64
end

rand(rng::AbstractRNG, ::Type{T}, s::ExponentialSampler) where {T} = T(s.scale) * randexp(rng, T)

struct ExponentialLogUSampler <: Sampleable{Univariate,Continuous}
    scale::Float64
end

function rand(rng::AbstractRNG, ::Type{T}, s::ExponentialLogUSampler) where {T}
    return -T(s.scale) * log(rand(rng, T))
end
