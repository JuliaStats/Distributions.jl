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

abstract type MultinomialSampler{T<:Real} <: Sampleable{Multivariate,Discrete} end

struct MultinomialSamplerBinomial{T<:Real} <: MultinomialSampler{T}
    n::Int
    k::Int
    prob::Vector{T}
end

struct MultinomialSamplerSequential{T<:Real} <: MultinomialSampler{T}
    n::Int
    k::Int
    alias::AliasTable
    scratch_idx::Vector{Int}
    scratch_acc::Vector{T}
end

function MultinomialSampler(n::Int, prob::Vector{<:Real})
    k = length(prob)

    # the constant λ term should be proportional to the perfomance ratio of
    # constructing and sampling from a Binomial vs sampling from an AliasTable
    λ = 10
    if n > λ * k
        MultinomialSamplerBinomial(n, k, prob)
    else
        MultinomialSamplerSequential(
            n,
            k,
            AliasTable(prob),
            Vector{Int}(undef, n),
            Vector{Float64}(undef, n)
        )
    end
end

function _rand!(rng::AbstractRNG, s::MultinomialSamplerBinomial, x::AbstractVector{<:Real})
    multinom_rand!(rng, s.n, s.prob, x)
end

function _rand!(rng::AbstractRNG, s::MultinomialSamplerSequential, x::AbstractVector{<:Real})
    fill!(x, zero(eltype(x)))
    at = s.alias
    rand!(rng, s.scratch_idx, 1:length(at.alias))
    rand!(rng, s.scratch_acc)

    @inbounds for i = 1:s.n
        i2 = s.scratch_idx[i] % Int
        x[ifelse(s.scratch_acc[i] < at.accept[i2], i2, at.alias[i2])] += 1
    end
    return x
end

length(s::MultinomialSampler) = s.k
