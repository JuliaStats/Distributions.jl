struct Generic{T<:Real,P<:Real} <: DiscreteUnivariateDistribution
    vals::Vector{T}
    probs::Vector{P}

    Generic{T,P}(vs::Vector{T}, ps::Vector{P}, ::NoArgCheck) where {T<:Real,P<:Real} =
        new(vs, ps)

    function Generic{T,P}(vs::Vector{T}, ps::Vector{P}) where {T<:Real,P<:Real}
        @check_args(Generic, isprobvec(ps))
        @check_args(Generic, allunique(vs))
        sort_order = sortperm(vs)
        new(vs[sort_order], ps[sort_order])
    end
end

support(d::Generic) = d.vals
probs(d::Generic)  = d.probs
params(d::Generic) = (d.vals, d.probs)

Generic(vs::Vector{T}, ps::Vector{P}) where {T<:Real, P<:Real} =
    Generic{T,P}(vs, ps)

rand(d::Generic{T,P}) where {T,P} =
    d.vals[searchsortedfirst(cumsum(d.probs), rand(P))]

# Helper functions for pdf and cdf required to fix ambiguous method
# error involving [pc]df(::DisceteUnivariateDistribution, ::Int)
function _pdf(d::Generic{T,P}, x::T) where {T,P}
    idx_range = searchsorted(d.vals, x)
    if length(idx_range) > 0
        return d.probs[first(idx_range)]
    else
        return zero(P)
    end
end
pdf(d::Generic{T}, x::Int) where T  = _pdf(d, convert(T, x))
pdf(d::Generic{T}, x::Real) where T = _pdf(d, convert(T, x))

_cdf(d::Generic{T}, x::T) where T =
    sum(d.probs[1:searchsortedlast(d.vals, x)])
cdf(d::Generic{T}, x::Int) where T = _cdf(d, convert(T, x))
cdf(d::Generic{T}, x::Real) where T = _cdf(d, convert(T, x))

quantile(d::Generic, q::Real) =
    d.vals[searchsortedfirst(cumsum(d.probs), q)]

minimum(d::Generic) = d.vals[1]
maximum(d::Generic) = d.vals[end]
insupport(d::Generic, x::Real) =
    length(searchsorted(d.vals, x)) > 0

mean(d::Generic) = dot(d.probs, d.vals)

function var(d::Generic{T}) where T
    m = mean(d)
    x, p = params(d)
    k = length(x)
    σ² = zero(T)
    for i in 1:k
        @inbounds σ² += abs2(x[i] - m) * p[i]
    end
    σ²
end

function skewness(d::Generic{T}) where T
    m = mean(d)
    x, p = params(d)
    k = length(x)
    μ₃ = zero(T)
    σ² = zero(T)
    for i in 1:k
        @inbounds d = x[i] - m
        @inbounds d²w = abs2(d) * p[i]
        μ₃ += d * d²w
        σ² += d²w
    end
    μ₃ / (σ² * sqrt(σ²))
end

function kurtosis(d::Generic{T}) where T
    m = mean(d)
    x, p = params(d)
    k = length(x)
    μ₄ = zero(T)
    σ² = zero(T)
    for i in 1:k
        @inbounds d² = abs2(x[i] - m)
        @inbounds d²w = d² * p[i]
        μ₄ += d² * d²w
        σ² += d²w
    end
    μ₄ / abs2(σ²) - 3
end

entropy(d::Generic) = entropy(d.probs)
entropy(d::Generic, b::Real) = entropy(d.probs, b)

mode(d::Generic) = d.vals[indmax(d.probs)]
function modes(d::Generic{T,P}) where {T,P}
    x, p = params(d)
    k = length(x)
    mds = T[]
    max_p = zero(P)
    for i in 1:k
        @inbounds pi = p[i]
        @inbounds xi = x[i]
        if pi > max_p
            max_p = pi
            mds = [xi]
        elseif pi == max_p
            push!(mds, xi)
        end
    end
    mds
end
