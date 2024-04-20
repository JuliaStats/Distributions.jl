function multinom_rand!(rng::AbstractRNG, n::Int, p::AbstractVector{<:Real},
                         x::AbstractVector{<:Real})
    k = length(p)
    length(x) == k || throw(DimensionMismatch("Invalid argument dimension."))

    z = zero(eltype(p))
    rp = oftype(z + z, 1) # remaining total probability (widens type if needed)
    i = 0
    km1 = k - 1

    while i < km1 && n > 0
        i += 1
        @inbounds pi = p[i]
        if pi < rp
            xi = rand(rng, Binomial(n, Float64(pi / rp)))
            @inbounds x[i] = xi
            n -= xi
            rp -= pi
        else
            # In this case, we don't even have to sample
            # from Binomial. Just assign remaining counts
            # to xi.

            @inbounds x[i] = n
            n = 0
            # rp = 0.0 (no need for this, as rp is no longer needed)
        end
    end

    if i == km1
        @inbounds x[k] = n
    else  # n must have been zero
        z = zero(eltype(x))
        for j = i+1 : k
            @inbounds x[j] = z
        end
    end

    return x
end

struct MultinomialSamplerBinomial{T<:Real} <: Sampleable{Multivariate,Discrete}
    n::Int
    k::Int
    prob::Vector{T}
end

struct MultinomialSamplerSequential{S} <: Sampleable{Multivariate,Discrete}
    n::Int
    k::Int
    alias::AliasTable
    scratch_alias_rng::Vector{S}
end

Base.@deprecate MultinomialSampler(n::Int, prob::Vector{<:Real}) MultinomialSamplerBinomial(n, length(prob), prob) false

function _rand!(rng::AbstractRNG, s::MultinomialSamplerBinomial, x::AbstractVector{<:Real})
    multinom_rand!(rng, s.n, s.prob, x)
end

function _rand!(rng::AbstractRNG, s::MultinomialSamplerSequential, x::AbstractVector{<:Real})
    fill!(x, zero(eltype(x)))
    rand!(rng, s.scratch_alias_rng)

    for r in s.scratch_alias_rng
        x[AliasTables.sample(r, s.alias.at)] += 1
    end
    return x
end

length(s::MultinomialSamplerBinomial) = s.k
length(s::MultinomialSamplerSequential) = s.k
