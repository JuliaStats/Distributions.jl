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

    function DiscreteNonParametric{T,P,Ts,Ps}(xs::Ts, ps::Ps; check_args::Bool=true) where {
            T<:Real,P<:Real,Ts<:AbstractVector{T},Ps<:AbstractVector{P}}
        check_args || return new{T,P,Ts,Ps}(xs, ps)
        @check_args(
            DiscreteNonParametric,
            (length(xs) == length(ps), "length of support and probability vector must be equal"),
            (ps, isprobvec(ps), "vector is not a probability vector"),
            (xs, allunique(xs), "support must contain only unique elements"),
        )
        sort_order = sortperm(xs)
        new{T,P,Ts,Ps}(xs[sort_order], ps[sort_order])
    end
end

DiscreteNonParametric(vs::AbstractVector{T}, ps::AbstractVector{P}; check_args::Bool=true) where {
        T<:Real,P<:Real} =
    DiscreteNonParametric{T,P,typeof(vs),typeof(ps)}(vs, ps; check_args=check_args)

Base.eltype(::Type{<:DiscreteNonParametric{T}}) where T = T

# Conversion
convert(::Type{DiscreteNonParametric{T,P,Ts,Ps}}, d::DiscreteNonParametric) where {T,P,Ts,Ps} =
    DiscreteNonParametric{T,P,Ts,Ps}(convert(Ts, support(d)), convert(Ps, probs(d)), check_args=false)
Base.convert(::Type{DiscreteNonParametric{T,P,Ts,Ps}}, d::DiscreteNonParametric{T,P,Ts,Ps}) where {T,P,Ts,Ps} = d

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

function Base.isapprox(c1::DiscreteNonParametric, c2::DiscreteNonParametric; kwargs...)
    support_c1 = support(c1)
    support_c2 = support(c2)
    return length(support_c1) == length(support_c2) &&
        isapprox(support_c1, support_c2; kwargs...) &&
        isapprox(probs(c1), probs(c2); kwargs...)
end

# Sampling

function rand(rng::AbstractRNG, d::DiscreteNonParametric)
    x = support(d)
    p = probs(d)
    n = length(p)
    draw = rand(rng, float(eltype(p)))
    cp = p[1]
    i = 1
    while cp <= draw && i < n
        @inbounds cp += p[i +=1]
    end
    return x[i]
end

sampler(d::DiscreteNonParametric) =
    DiscreteNonParametricSampler(support(d), probs(d))

# Override the method in testutils.jl since it assumes
# an evenly-spaced integer support
get_evalsamples(d::DiscreteNonParametric, ::Float64) = support(d)

# Evaluation

pdf(d::DiscreteNonParametric) = copy(probs(d))

function pdf(d::DiscreteNonParametric, x::Real)
    s = support(d)
    idx = searchsortedfirst(s, x)
    ps = probs(d)
    if idx <= length(ps) && s[idx] == x
        return ps[idx]
    else
        return zero(eltype(ps))
    end
end
logpdf(d::DiscreteNonParametric, x::Real) = log(pdf(d, x))

function cdf(d::DiscreteNonParametric, x::Real)
    ps = probs(d)
    P = float(eltype(ps))

    # trivial cases
    x < minimum(d) && return zero(P)
    x >= maximum(d) && return one(P)
    isnan(x) && return P(NaN)

    n = length(ps)
    stop_idx = searchsortedlast(support(d), x)
    s = zero(P)
    if stop_idx < div(n, 2)
        @inbounds for i in 1:stop_idx
            s += ps[i]
        end
    else
        @inbounds for i in (stop_idx + 1):n
            s += ps[i]
        end
        s = 1 - s
    end

    return s
end

function ccdf(d::DiscreteNonParametric, x::Real)
    ps = probs(d)
    P = float(eltype(ps))

    # trivial cases
    x < minimum(d) && return one(P)
    x >= maximum(d) && return zero(P)
    isnan(x) && return P(NaN)

    n = length(ps)
    stop_idx = searchsortedlast(support(d), x)
    s = zero(P)
    if stop_idx < div(n, 2)
        @inbounds for i in 1:stop_idx
            s += ps[i]
        end
        s = 1 - s
    else
        @inbounds for i in (stop_idx + 1):n
            s += ps[i]
        end
    end

    return s
end

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

function var(d::DiscreteNonParametric)
    x = support(d)
    p = probs(d)
    return var(x, Weights(p, one(eltype(p))); corrected=false)
end

function skewness(d::DiscreteNonParametric)
    x = support(d)
    p = probs(d)
    return skewness(x, Weights(p, one(eltype(p))))
end

function kurtosis(d::DiscreteNonParametric)
    x = support(d)
    p = probs(d)
    return kurtosis(x, Weights(p, one(eltype(p))))
end

entropy(d::DiscreteNonParametric) = entropy(probs(d))
entropy(d::DiscreteNonParametric, b::Real) = entropy(probs(d), b)

function mode(d::DiscreteNonParametric)
    x = support(d)
    p = probs(d)
    return mode(x, Weights(p, one(eltype(p))))
end
function modes(d::DiscreteNonParametric)
    x = support(d)
    p = probs(d)
    return modes(x, Weights(p, one(eltype(p))))
end

function mgf(d::DiscreteNonParametric, t::Real)
    x = support(d)
    p = probs(d)
    s = zero(Float64)
    for i in 1:length(x)
        s += p[i] * exp(t*x[i])
    end
    s
end

function cf(d::DiscreteNonParametric, t::Real)
    x = support(d)
    p = probs(d)
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
                   w::AbstractArray{W}; check_args::Bool=true) where {T<:Real,W<:Real}

    @check_args DiscreteNonParametric (length(x) == length(w))

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
    DiscreteNonParametric{T,W,Ts,Ws}(ss.support, normalize!(copy(ss.freq), 1), check_args=false)
