##### Alias Table #####

immutable AliasTable <: Sampleable{Univariate,Discrete}
    accept::Vector{Float64}
    alias::Vector{Int}
    isampler::RandIntSampler
end
numcategories(s::AliasTable) = length(s.accept)

function make_alias_table!(a::AbstractVector{Float64})
    # input probabilities via a, which is then
    # overriden by acceptance probabilities
    # and used internally by the AliasTable instance
    #
    n = length(a)
    for i=1:n
        @inbounds a[i] *= n
    end
    alias = Array(Int,n)
    larges = Int[]
    smalls = Int[]

    for i = 1:n
        acci = a[i] 
        if acci > 1.0 
            push!(larges,i)
        elseif acci < 1.0
            push!(smalls,i)
        end
    end

    while !isempty(larges) && !isempty(smalls)
        s = pop!(smalls)
        l = pop!(larges)
        @inbounds alias[s] = l
        @inbounds a[l] = (a[l] - 1.0) + a[s]
        if a[l] > 1
            push!(larges,l)
        else
            push!(smalls,l)
        end
    end

    # this loop should be redundant, except for rounding
    for s in smalls
        @inbounds a[s] = 1.0
    end
    AliasTable(a, alias, RandIntSampler(n))
end

function AliasTable{T<:Real}(probs::AbstractVector{T})
    n = length(probs)
    n > 0 || error("The input probability vector is empty.")
    a = Array(Float64, n)
    for i = 1:n
        @inbounds a[i] = probs[i]
    end
    make_alias_table!(a)
end

rand(s::AliasTable) = 
    (i = rand(s.isampler); u = rand(); @inbounds r = u < s.accept[i] ? i : s.alias[i]; r)

show(io::IO, s::AliasTable) = @printf(io, "AliasTable with %d entries", numcategories(s))

