function mvhypergeom_rand!(rng::AbstractRNG, m::AbstractVector{Int}, n::Int,
    x::AbstractVector{<:Real})
    k = length(m)
    length(x) == k || throw(DimensionMismatch("Invalid argument dimension."))

    M = sum(m)
    i = 0
    km1 = k - 1

    while i < km1 && n > 0
        i += 1
        mi = m[i]
        # Sample from hypergeometric distribution. Element of type i are 
        # considered successes. All other elements are considered failures.
        xi = rand(rng, Hypergeometric(mi, M - mi, n))
        x[i] = xi
        # Remove elements of type i from population and group to be sampled.
        n -= xi
        M -= mi
    end

    if i == km1
        x[k] = n
    else  # n must have been zero.
        z = zero(eltype(x))
        for j = i+1:k
            x[j] = z
        end
    end

    return x
end
