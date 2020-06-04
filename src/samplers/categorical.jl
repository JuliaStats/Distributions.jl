#### naive sampling

struct CategoricalDirectSampler{T<:Real,Ts<:AbstractVector{T}} <: Sampleable{Univariate,Discrete}
    prob::Ts

    function CategoricalDirectSampler{T,Ts}(p::Ts) where {
        T<:Real,Ts<:AbstractVector{T}}
        isempty(p) && throw(ArgumentError("p is empty."))
        new{T,Ts}(p)
    end
end

CategoricalDirectSampler(p::Ts) where {T<:Real,Ts<:AbstractVector{T}} =
    CategoricalDirectSampler{T,Ts}(p)

ncategories(s::CategoricalDirectSampler) = length(s.prob)

function rand(rng::AbstractRNG, s::CategoricalDirectSampler)
    p = s.prob
    n = length(p)
    i = 1
    c = p[1]
    u = rand(rng)
    while c < u && i < n
        c += p[i += 1]
    end
    return i
end
