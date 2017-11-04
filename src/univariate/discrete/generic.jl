struct Generic{T<:Real,P<:Real,S<:AbstractVector{T}} <: DiscreteUnivariateDistribution
    support::S
    p::Vector{P}

    Generic{T,P,S}(vs::S, ps::Vector{P}, ::NoArgCheck) where {T<:Real,P<:Real,S<:AbstractVector{T}} =
        new(vs, ps)

    function Generic{T,P,S}(vs::S, ps::Vector{P}) where {T<:Real,P<:Real,S<:AbstractVector{T}}
        @check_args(Generic, isprobvec(ps))
        @check_args(Generic, allunique(vs))
        sort_order = sortperm(vs)
        new(vs[sort_order], ps[sort_order])
    end
end

Generic(vs::S, ps::Vector{P}) where {T<:Real,P<:Real,S<:AbstractVector{T}} =
    Generic{T,P,S}(vs, ps)

# Conversion
convert(::Type{Generic{T,P,S}}, d::Generic) where {T,P,S} =
    Generic{T,P,S}(S(support(d)), Vector{P}(probs(d)), NoArgCheck())

params(d::Generic) = (d.support, d.p)
support(d::Generic) = d.support
probs(d::Generic)  = d.p

rand(d::Generic{T,P}) where {T,P} =
    support(d)[searchsortedfirst(cumsum(probs(d)), rand(P))] #TODO: Switch to single pass

# Evaluation

pdf(d::Generic) = copy(probs(d))

# Helper functions for pdf and cdf required to fix ambiguous method
# error involving [pc]df(::DisceteUnivariateDistribution, ::Int)
function _pdf(d::Generic{T,P}, x::T) where {T,P}
    idx_range = searchsorted(support(d), x)
    if length(idx_range) > 0
        return probs(d)[first(idx_range)]
    else
        return zero(P)
    end
end
pdf(d::Generic{T}, x::Int) where T  = _pdf(d, convert(T, x))
pdf(d::Generic{T}, x::Real) where T = _pdf(d, convert(T, x))

_cdf(d::Generic{T}, x::T) where T =
    sum(probs(d)[1:searchsortedlast(support(d), x)]) #TODO: Switch to single-pass
cdf(d::Generic{T}, x::Int) where T = _cdf(d, convert(T, x))
cdf(d::Generic{T}, x::Real) where T = _cdf(d, convert(T, x))

function quantile(d::Generic, q::Real)
    0 <= q <= 1 || throw(DomainError())
    x = support(d)
    p = probs(d)
    k = length(x)
    i = 1
    cp = p[1]
    while cp < q && i < k #Note: is i < k necessary?
        i += 1
        @inbounds cp += p[i]
    end
    x[i]
end


minimum(d::Generic) = support(d)[1]
maximum(d::Generic) = support(d)[end]
insupport(d::Generic, x::Real) =
    length(searchsorted(support(d), x)) > 0

mean(d::Generic) = dot(probs(d), support(d))

function var(d::Generic{T}) where T
    m = mean(d)
    x = support(d)
    p = probs(d)
    k = length(x)
    σ² = zero(T)
    for i in 1:k
        @inbounds σ² += abs2(x[i] - m) * p[i]
    end
    σ²
end

function skewness(d::Generic{T}) where T
    m = mean(d)
    x = support(d)
    p = probs(d)
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
    x = support(d)
    p = probs(d)
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

entropy(d::Generic) = entropy(probs(d))
entropy(d::Generic, b::Real) = entropy(probs(d), b)

mode(d::Generic) = support(d)[indmax(probs(d))]
function modes(d::Generic{T,P}) where {T,P}
    x = support(d)
    p = probs(d)
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

function mgf(d::Generic, t::Real)
    x, p = params(d)
    dot(p, exp.(t.*x))
end

function cf(d::Generic, t::Real)
    x, p = params(d)
    dot(p, cis.(t.*x))
end

# TODO: Sufficient statistics

# TODO: MLE
