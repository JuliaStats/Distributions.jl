"""
A small helper struct representing an identity permutation.

The only supported operation is `IdentityPermute()[x]` -> x for all x.
"""
struct IdentityPermute; end
Base.getindex(id::IdentityPermute, idx::Int) = idx

"""
    InversionCDT{T,P} <: Sampleable{Univariate,Discrete}

The `InversionCDT` sampler supports sampling from arbitrary discrete probability
distributions using inversion sampling. The sampler stores a table of
cummulative probabilities and then samples a value u ∈ [0,1) uniformly at
random, returning the first index whose cummulative probability weight is larger
than the sampled value.

If the provided probabilities are of floating point, type,
the default constructor sorts the provided probabilities to take advantage of
increased floating point precision near 0.0 (as suggested in [MW19]).
The obtained permutation of probabilities is kept internally and transparent
to the user. This optimization may be bypassed by using the
`InversionCDT{T, IdentityPermute}` constructor directly.

For n, the size of the support of the discrete distribution (equivalently the
length of the provided probability vector), the cost to sample a value is
O(log(n)), while the preprocessing cost is either O(n) or O(n log(n)) depending
on whether or not the sorting optimization is enabled.

[MW19] Michael Walter, "Sampling the Integers with Low Relative Error"
      (https://eprint.iacr.org/2019/068).
"""
struct InversionCDT{T<:AbstractVector,P} <: Sampleable{Univariate,Discrete}
    table::T
    permute::P
end
ncategories(s::InversionCDT) = length(s.table)

InversionCDT(probs::T) where {T <: AbstractVector} = InversionCDT{T, IdentityPermute}(probs)

function InversionCDT(probs::T) where {S <: AbstractFloat, T <: AbstractVector{S}}
    # Floating point numbers have more precision near zero. As a result,
    # sorting the probabilities before constructing the cdt table. See [MW19]
    # Lemma 3.
    perm = sortperm(probs)
    InversionCDT{T, Vector{Int}}(cumsum(probs[perm][1:end-1]), perm)
end

function rand(rng::AbstractRNG, cdt::InversionCDT{<:AbstractVector{T}}) where {T}
    u = rand(rng, T)
    @assert zero(T) <= u <= one(T)
    @inbounds cdt.permute[searchsortedfirst(cdt.table, u)]
end

