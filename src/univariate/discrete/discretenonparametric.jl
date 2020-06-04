"""
    DiscreteNonParametric(xs, ps)

A *Discrete nonparametric distribution* explicitly defines an arbitrary
probability mass function in terms of a list of real support values and their
corresponding probabilities

```julia
d = DiscreteNonParametric(xs, ps)

params(d)  # Get the parameters, i.e. (xs, ps)
support(d) # Get a sorted AbstractVector describing the support (xs) of the distribution
probs(d)   # Get a Vector of the probabilities (ps) associated with the support
```

External links

* [Probability mass function on Wikipedia](http://en.wikipedia.org/wiki/Probability_mass_function)
"""
struct DiscreteNonParametric{T<:Real,P<:Real,Ts<:AbstractVector{T},Ps<:AbstractVector{P}} <: DiscreteUnivariateDistribution
    support::Ts
    p::Ps

    function DiscreteNonParametric{T,P,Ts,Ps}(vs::Ts, ps::Ps; check_args=true) where {
            T<:Real,P<:Real,Ts<:AbstractVector{T},Ps<:AbstractVector{P}}
        check_args || return new{T,P,Ts,Ps}(vs, ps)
        @check_args(DiscreteNonParametric, length(vs) == length(ps))
        @check_args(DiscreteNonParametric, isprobvec(ps))
        @check_args(DiscreteNonParametric, allunique(vs))
        sort_order = sortperm(vs)
        new{T,P,Ts,Ps}(vs[sort_order], ps[sort_order])
    end
end

DiscreteNonParametric(vs::Ts, ps::Ps; check_args=true) where {
        T<:Real,P<:Real,Ts<:AbstractVector{T},Ps<:AbstractVector{P}} =
    DiscreteNonParametric{T,P,Ts,Ps}(vs, ps, check_args=check_args)

Base.eltype(::Type{<:DiscreteNonParametric{T}}) where T = T

# Conversion
convert(::Type{DiscreteNonParametric{T,P,Ts,Ps}}, d::DiscreteNonParametric) where {T,P,Ts,Ps} =
    DiscreteNonParametric{T,P,Ts,Ps}(Ts(support(d)), Ps(probs(d)), check_args=false)

# Accessors
params(d::DiscreteNonParametric) = (d.support, d.p)

"""
    support(d::DiscreteNonParametric)

Get a sorted AbstractVector defining the support of `d`.
"""
support(d::DiscreteNonParametric) = d.support

"""
    probs(d::DiscreteNonParametric)

Get the vector of probabilities associated with the support of `d`.
"""
probs(d::DiscreteNonParametric)  = d.p

==(c1::D, c2::D) where D<:DiscreteNonParametric =
    (support(c1) == support(c2) || all(support(c1) .== support(c2))) &&
    (probs(c1) == probs(c2) || all(probs(c1) .== probs(c2)))

Base.isapprox(c1::D, c2::D) where D<:DiscreteNonParametric =
    (support(c1) ≈ support(c2) || all(support(c1) .≈ support(c2))) &&
    (probs(c1) ≈ probs(c2) || all(probs(c1) .≈ probs(c2)))

# Sampling

function rand(rng::AbstractRNG, d::DiscreteNonParametric{T,P}) where {T,P}
    x = support(d)
    p = probs(d)
    n = length(p)
    draw = rand(rng, P)
    cp = zero(P)
    i = 0
    while cp < draw && i < n
        cp += p[i +=1]
    end
    x[max(i,1)]
end

rand(d::DiscreteNonParametric) = rand(GLOBAL_RNG, d)

sampler(d::DiscreteNonParametric) =
    DiscreteNonParametricSampler(support(d), probs(d))

# Override the method in testutils.jl since it assumes
# an evenly-spaced integer support
get_evalsamples(d::DiscreteNonParametric, ::Float64) = support(d)

# Evaluation

pdf(d::DiscreteNonParametric) = copy(probs(d))

# Helper functions for pdf and cdf required to fix ambiguous method
# error involving [pc]df(::DisceteUnivariateDistribution, ::Int)
function _pdf(d::DiscreteNonParametric{T,P}, x::T) where {T,P}
    idx_range = searchsorted(support(d), x)
    if length(idx_range) > 0
        return probs(d)[first(idx_range)]
    else
        return zero(P)
    end
end
pdf(d::DiscreteNonParametric{T}, x::Int) where T  = _pdf(d, convert(T, x))
pdf(d::DiscreteNonParametric{T}, x::Real) where T = _pdf(d, convert(T, x))

function _cdf(d::DiscreteNonParametric{T,P}, x::T) where {T,P}
    x > maximum(d) && return 1.0
    s = zero(P)
    ps = probs(d)
    stop_idx = searchsortedlast(support(d), x)
    for i in 1:stop_idx
        s += ps[i]
    end
    return s
end
cdf(d::DiscreteNonParametric{T}, x::Integer) where T = _cdf(d, convert(T, x))
cdf(d::DiscreteNonParametric{T}, x::Real) where T = _cdf(d, convert(T, x))

function _ccdf(d::DiscreteNonParametric{T,P}, x::T) where {T,P}
    x < minimum(d) && return 1.0
    s = zero(P)
    ps = probs(d)
    stop_idx = searchsortedlast(support(d), x)
    for i in (stop_idx+1):length(ps)
        s += ps[i]
    end
    return s
end
ccdf(d::DiscreteNonParametric{T}, x::Integer) where T = _ccdf(d, convert(T, x))
ccdf(d::DiscreteNonParametric{T}, x::Real) where T = _ccdf(d, convert(T, x))

function quantile(d::DiscreteNonParametric, q::Real)
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

minimum(d::DiscreteNonParametric) = first(support(d))
maximum(d::DiscreteNonParametric) = last(support(d))
insupport(d::DiscreteNonParametric, x::Real) =
    length(searchsorted(support(d), x)) > 0

mean(d::DiscreteNonParametric) = dot(probs(d), support(d))

function var(d::DiscreteNonParametric{T}) where T
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

function skewness(d::DiscreteNonParametric{T}) where T
    m = mean(d)
    x = support(d)
    p = probs(d)
    k = length(x)
    μ₃ = zero(T)
    σ² = zero(T)
    @inbounds for i in 1:k
        d = x[i] - m
        d²w = abs2(d) * p[i]
        μ₃ += d * d²w
        σ² += d²w
    end
    μ₃ / (σ² * sqrt(σ²))
end

function kurtosis(d::DiscreteNonParametric{T}) where T
    m = mean(d)
    x = support(d)
    p = probs(d)
    k = length(x)
    μ₄ = zero(T)
    σ² = zero(T)
    @inbounds for i in 1:k
        d² = abs2(x[i] - m)
        d²w = d² * p[i]
        μ₄ += d² * d²w
        σ² += d²w
    end
    μ₄ / abs2(σ²) - 3
end

entropy(d::DiscreteNonParametric) = entropy(probs(d))
entropy(d::DiscreteNonParametric, b::Real) = entropy(probs(d), b)

mode(d::DiscreteNonParametric) = support(d)[argmax(probs(d))]
function modes(d::DiscreteNonParametric{T,P}) where {T,P}
    x = support(d)
    p = probs(d)
    k = length(x)
    mds = T[]
    max_p = zero(P)
    @inbounds for i in 1:k
        pi = p[i]
        xi = x[i]
        if pi > max_p
            max_p = pi
            mds = [xi]
        elseif pi == max_p
            push!(mds, xi)
        end
    end
    mds
end

function mgf(d::DiscreteNonParametric, t::Real)
    x, p = params(d)
    s = zero(Float64)
    for i in 1:length(x)
        s += p[i] * exp(t*x[i])
    end
    s
end

function cf(d::DiscreteNonParametric, t::Real)
    x, p = params(d)
    s = zero(Complex{Float64})
    for i in 1:length(x)
       s += p[i] * cis(t*x[i])
    end
    s
end

# Sufficient statistics

struct DiscreteNonParametricStats{T<:Real,W<:Real,Ts<:AbstractVector{T},
                                  Ws<:AbstractVector{W}} <: SufficientStats
    support::Ts
    freq::Ws
end

function suffstats(::Type{<:DiscreteNonParametric}, x::AbstractArray{T}) where {T<:Real}

    N = length(x)
    N == 0 && return DiscreteNonParametricStats(T[], Float64[])

    n = 1
    vs = Vector{T}(undef,N)
    ps = zeros(Float64, N)
    x = sort(vec(x))

    vs[1] = x[1]
    ps[1] += 1.

    xprev = x[1]
    @inbounds for i = 2:N
        xi = x[i]
        if xi != xprev
            n += 1
            vs[n] = xi
        end
        ps[n] += 1.
        xprev = xi
    end

    resize!(vs, n)
    resize!(ps, n)
    DiscreteNonParametricStats(vs, ps)

end

function suffstats(::Type{<:DiscreteNonParametric}, x::AbstractArray{T},
                   w::AbstractArray{W}) where {T<:Real,W<:Real}

    @check_args(DiscreteNonParametric, length(x) == length(w))

    N = length(x)
    N == 0 && return DiscreteNonParametricStats(T[], W[])

    n = 1
    vs = Vector{T}(undef, N)
    ps = zeros(W, N)

    xorder = sortperm(vec(x))
    x = vec(x)[xorder]
    w = vec(w)[xorder]

    vs[1] = x[1]
    ps[1] += w[1]

    xprev = x[1]
    @inbounds for i = 2:N
        xi = x[i]
        wi = w[i]
        if xi != xprev
            n += 1
            vs[n] = xi
        end
        ps[n] += wi
        xprev = xi
    end

    resize!(vs, n)
    resize!(ps, n)
    DiscreteNonParametricStats(vs, ps)

end

# # Model fitting

fit_mle(::Type{<:DiscreteNonParametric},
        ss::DiscreteNonParametricStats{T,W,Ts,Ws}) where {T,W,Ts,Ws} =
    DiscreteNonParametric{T,W,Ts,Ws}(ss.support, pnormalize!(copy(ss.freq)), check_args=false)
