function mvhypergeom_rand!(rng::AbstractRNG, m::AbstractVector{Int}, n::Int,
    x::AbstractVector{<:Real})
        k = length(m)
        length(x) == k || throw(DimensionMismatch("Invalid argument dimension."))

        M = sum(m) # remaining population
        i = 0
        km1 = k - 1

        while i < km1 && n > 0
            i += 1
            @inbounds mi = m[i]
            # Sample from hypergeometric distribution 
            # element of type i are considered successes
            # all other elements are considered failures
            xi = rand(rng, Hypergeometric(mi, M-mi, n)) 
            @inbounds x[i] = xi
            # Remove elements of type i from population
            n -= xi
            M -= mi      
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
