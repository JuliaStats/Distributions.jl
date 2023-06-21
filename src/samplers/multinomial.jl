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

struct MultinomialSampler{T<:Real} <: Sampleable{Multivariate,Discrete}
    n::Int
    prob::Vector{T}
    alias::AliasTable
end

function MultinomialSampler(n::Int, prob::Vector{<:Real})
    return MultinomialSampler(n, prob, AliasTable(prob))
end

function _rand!(rng::AbstractRNG, s::MultinomialSampler,
                x::AbstractVector{<:Real})
    n = s.n
    k = length(s)
    if n^2 > k
        multinom_rand!(rng, n, s.prob, x)
    else
        # Use an alias table
        fill!(x, zero(eltype(x)))
        a = s.alias
        for i = 1:n
            x[rand(rng, a)] += 1
        end
    end
    return x
end

length(s::MultinomialSampler) = length(s.prob)
