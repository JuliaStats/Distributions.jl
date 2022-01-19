"""
    censored(d0::UnivariateDistribution, l::Union{Real,Missing}, u::Union{Real,Missing})

A _censored distribution_ `d` of a distribution `d0` to the interval `[l, u]` has the
probability density (mass) function:

```math
f(x; d_0, l, u) = \\begin{cases}
    P_{Z \\sim d_0}(Z \\le l), & x = l \\\\
    f_{d_0}(x),              & l < x < u \\\\
    P_{Z \\sim d_0}(Z \\ge u), & x = u \\\\
  \\end{cases}, \\quad x \\in [l, u]
```
where ``f_{d_0}`` is the probability density (mass) function of ``d_0``.

If ``Z`` is a variate from ``d_0``, and `X = clamp(Z, l, u)`, then ``X`` is a variate from
``d``, the censored version of ``d_0``. Note that this implies that even if ``d`` is
continuous, its censored form assigns positive probability to the bounds ``l`` and `u``.
Therefore a censored continuous distribution has atoms and is a mixture of discrete and
continuous components.

```julia
censored(d0, l, missing)   # d0 left-censored to the interval [l, Inf)
censored(d0, missing, u)   # d0 right-censored to the interval (-Inf, u]
censored(d0, l, u)         # d0 interval-censored to the interval [l, u]
```

The function falls back to constructing a [`Censored`](@ref) wrapper.

# Implementation

To implement a specialized censored form for distributions of type `D`, one or more of the
following methods should be implemented:
- `censored(d::D, l::T, u::T) where {T <: Real}`
- `censored(d::D, ::Missing, u::Real)`
- `censored(d::D, l::Real, ::Missing)`
"""
censored
function censored(d::UnivariateDistribution, l::T, u::T) where {T<:Real}
    return Censored(d, l, u)
end
function censored(d::UnivariateDistribution, ::Missing, u::Real)
    return Censored(d, missing, u)
end
function censored(d::UnivariateDistribution, l::Real, ::Missing)
    return Censored(d, l, missing)
end
censored(d::UnivariateDistribution, l::Real, u::Real) = censored(d, promote(l, u)...)
censored(d::UnivariateDistribution, ::Missing, ::Missing) = d

"""
    Censored

Generic wrapper for a [`censored`](@ref) distribution.
"""
struct Censored{
    D<:UnivariateDistribution,
    S<:ValueSupport,
    T<:Real,
    TL<:Union{T,Missing},
    TU<:Union{T,Missing},
} <: UnivariateDistribution{S}
    uncensored::D      # the original distribution (uncensored)
    lower::TL      # lower bound
    upper::TU      # upper bound
    function Censored(d::UnivariateDistribution, lower::T, upper::T; check_args::Bool=true) where {T<:Real}
        check_args && @check_args(Censored, lower ≤ upper) 
        new{typeof(d), value_support(typeof(d)), T, T, T}(d, lower, upper)
    end
    function Censored(d::UnivariateDistribution, l::Missing, u::Real; check_args::Bool=true)
        new{typeof(d), value_support(typeof(d)), typeof(u), Missing, typeof(u)}(d, l, u)
    end
    function Censored(d::UnivariateDistribution, l::Real, u::Missing; check_args::Bool=true)
        new{typeof(d), value_support(typeof(d)), typeof(l), typeof(l), Missing}(d, l, u)
    end
end

function censored(d::Censored, l::T, u::T) where {T<:Real}
    return censored(
        d.uncensored,
        d.lower === missing ? l : max(l, d.lower),
        d.upper === missing ? u : min(u, d.upper),
    )
end
function censored(d::Censored, ::Missing, u::Real)
    return censored(d.uncensored, d.lower, d.upper === missing ? u : min(u, d.upper))
end
function censored(d::Censored, l::Real, ::Missing)
    return censored(d.uncensored, d.lower === missing ? l : max(l, d.lower), d.upper)
end

function params(d::Censored)
    d0params = params(d.uncensored)
    return (d0params..., d.lower, d.upper)
end

function partype(d::Censored{<:UnivariateDistribution,<:ValueSupport,T}) where {T}
    return promote_type(partype(d.uncensored), T)
end

Base.eltype(::Type{<:Censored{D,S,T}}) where {D,S,T} = promote_type(T, eltype(D))

#### Range and Support

function islowerbounded(d::Censored)
    return (
        islowerbounded(d.uncensored) ||
        (d.lower !== missing && cdf(d.uncensored, d.lower) > 0)
    )
end
function isupperbounded(d::Censored)
    return (
        isupperbounded(d.uncensored) ||
        (d.upper !== missing && _ccdf_inclusive(d.uncensored, d.upper) > 0)
    )
end

function minimum(d::Censored)
    d0min = minimum(d.uncensored)
    return d.lower === missing ? min(d0min, d.upper) : max(d0min, d.lower)
end

function maximum(d::Censored)
    d0max = maximum(d.uncensored)
    return d.upper === missing ? max(d0max, d.lower) : min(d0max, d.upper)
end

function insupport(d::Censored{<:UnivariateDistribution}, x::Real)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    return (
        _in_open_interval(x, lower, upper) ||
        (_eqnotmissing(x, lower) && cdf(d0, lower) > 0) ||
        (_eqnotmissing(x, upper) && _ccdf_inclusive(d0, upper) > 0)
    )
end

#### Show

function show(io::IO, ::MIME"text/plain", d::Censored)
    print(io, "Censored(")
    d0 = d.uncensored
    uml, namevals = _use_multline_show(d0)
    uml ? show_multline(io, d0, namevals; newline=false) : show_oneline(io, d0, namevals)
    print(io, ", range=(", d.lower, ", ", d.upper, "))")
end

_use_multline_show(d::Censored) = _use_multline_show(d.uncensored)


#### Statistics

quantile(d::Censored, p::Real) = _clamp(quantile(d.uncensored, p), d.lower, d.upper)

median(d::Censored) = _clamp(median(d.uncensored), d.lower, d.upper)

# the expectations use the following relation:
# 𝔼_{x ~ d}[h(x)] = P_{x ~ d₀}(x < l) h(l) + P_{x ~ d₀}(x > u) h(u)
#                 + P_{x ~ d₀}(l ≤ x ≤ u) 𝔼_{x ~ τ}[h(x)],
# where d₀ is the uncensored distribution, d is d₀ censored to [l, u],
# and τ is d₀ truncated to [l, u]

function mean(d::Censored)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    if lower === missing
        log_prob_lower = -Inf
        prob_lower = 0
    else
        log_prob_lower = _logcdf_noninclusive(d0, lower)
        prob_lower = exp(log_prob_lower)
    end
    if upper === missing
        log_prob_upper = -Inf
        prob_upper = 0
    else
        log_prob_upper = logccdf(d0, upper)
        prob_upper = exp(log_prob_upper)
    end
    prob_trunc = exp(log1mexp(logaddexp(log_prob_lower, log_prob_upper)))
    if prob_trunc < eps(one(prob_trunc)) # truncation contains ~ no probability
        if lower === missing
            return one(prob_trunc) * upper
        elseif upper === missing
            return one(prob_trunc) * lower
        else
            return prob_lower * (iszero(prob_lower) ? oneunit(lower) : lower) +
                   prob_upper * (iszero(prob_upper) ? oneunit(upper) : upper)
        end
    end
    dtrunc = _to_truncated(d)
    μ = prob_trunc * mean(dtrunc)
    if !iszero(prob_lower)
        μ += prob_lower * lower
    end
    if !iszero(prob_upper)
        μ += prob_upper * upper
    end
    return μ
end

function var(d::Censored)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    if lower === missing
        log_prob_lower = -Inf
        prob_lower = 0
    else
        log_prob_lower = _logcdf_noninclusive(d0, lower)
        prob_lower = exp(log_prob_lower)
    end
    if upper === missing
        log_prob_upper = -Inf
        prob_upper = 0
    else
        log_prob_upper = logccdf(d0, upper)
        prob_upper = exp(log_prob_upper)
    end
    prob_trunc = exp(log1mexp(logaddexp(log_prob_lower, log_prob_upper)))
    if prob_trunc < eps(one(prob_trunc)) # truncation contains ~ no probability
        if lower === missing
            return one(prob_trunc) * abs2(zero(upper))
        elseif upper === missing
            return one(prob_trunc) * abs2(zero(lower))
        else
            μ = prob_lower * (iszero(prob_lower) ? oneunit(lower) : lower) +
                prob_upper * (iszero(prob_upper) ? oneunit(upper) : upper)
            v = prob_lower * abs2(iszero(prob_lower) ? oneunit(lower) : lower - μ) +
                prob_upper * abs2(iszero(prob_upper) ? oneunit(upper) : upper - μ)
            return v
        end
    end

    dtrunc = _to_truncated(d)
    μ_trunc = mean(dtrunc)
    μ = prob_trunc * μ_trunc
    if !iszero(prob_lower)
        μ += prob_lower * lower
    end
    if !iszero(prob_upper)
        μ += prob_upper * upper
    end

    v = prob_trunc * (var(dtrunc) + abs2(μ_trunc - μ))
    if !iszero(prob_lower)
        v += prob_lower * abs2(lower - μ)
    end
    if !iszero(prob_upper)
        v += prob_upper * abs2(upper - μ)
    end
    return v
end

# this expectation also uses the following relation:
# 𝔼_{x ~ τ}[-log d(x)] = S[τ] - log P_{x ~ d₀}(l ≤ x ≤ u)
#   + (P_{x ~ d₀}(x = l) (log P_{x ~ d₀}(x = l) - log P_{x ~ d₀}(x ≤ l)) + 
#      P_{x ~ d₀}(x = u) (log P_{x ~ d₀}(x = u) - log P_{x ~ d₀}(x ≥ u))
#   ) / P_{x ~ d₀}(l ≤ x ≤ u),
# where S[τ] is the entropy of τ.

function entropy(d::Censored)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    if lower === missing
        log_prob_upper = logccdf(d0, upper)
        if value_support(typeof(d0)) === Discrete
            logpu = logpdf(d0, upper)
            log_prob_upper_inc = logaddexp(log_prob_upper, logpu)
            xlogx_pu = xexpx(logpu)
        else
            log_prob_upper_inc = log_prob_upper
            xlogx_pu = 0
        end
        entropy_bound = -xexpx(log_prob_upper_inc)
        log_prob_trunc = log1mexp(log_prob_upper)    
        xlogx_pl = 0
    elseif upper === missing
        log_prob_lower_inc = logcdf(d0, lower)
        if value_support(typeof(d0)) === Discrete
            logpl = logpdf(d0, lower)
            log_prob_lower = logsubexp(log_prob_lower_inc, logpl)
            xlogx_pl = xexpx(logpl)
        else
            log_prob_lower = log_prob_lower_inc
            xlogx_pl = 0
        end
        entropy_bound = -xexpx(log_prob_lower_inc)
        log_prob_trunc = log1mexp(log_prob_lower)    
        xlogx_pu = 0
    else
        log_prob_lower_inc = logcdf(d0, lower)
        log_prob_upper = logccdf(d0, upper)
        if value_support(typeof(d0)) === Discrete
            logpl = logpdf(d0, lower)
            logpu = logpdf(d0, upper)
            log_prob_lower = logsubexp(log_prob_lower_inc, logpl)
            log_prob_upper_inc = logaddexp(log_prob_upper, logpu)
            xlogx_pl = xexpx(logpl)
            xlogx_pu = xexpx(logpu)
        else
            log_prob_lower = log_prob_lower_inc
            log_prob_upper_inc = log_prob_upper
            xlogx_pl = xlogx_pu = 0
        end
        entropy_bound = -(xexpx(log_prob_lower_inc) + xexpx(log_prob_upper_inc))
        log_prob_trunc = log1mexp(logaddexp(log_prob_lower, log_prob_upper))
    end
    
    # truncation contains ~ no probability
    if log_prob_trunc < log(eps(one(log_prob_trunc)))
        return entropy_bound
    end

    dtrunc = _to_truncated(d)
    entropy_interval = 
        exp(log_prob_trunc) * entropy(dtrunc) - xexpx(log_prob_trunc) + xlogx_pl + xlogx_pu
    return entropy_bound + entropy_interval
end


#### Evaluation

function pdf(d::Censored{<:Any,<:Any,T}, x::Real) where {T}
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    S = Base.promote_eltype(T, x)
    return if lower !== missing && x == lower
        result = cdf(d0, S(x))
        _eqnotmissing(x, upper) ? one(result) : result
    elseif _eqnotmissing(x, upper)
        _ccdf_inclusive(d0, S(x))
    else
        result = pdf(d0, S(x))
        _in_open_interval(x, lower, upper) ? result : zero(result)
    end
end

function logpdf(d::Censored{<:Any,<:Any,T}, x::Real) where {T}
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    S = Base.promote_eltype(T, x)
    return if lower !== missing && x == lower
        result = logcdf(d0, S(x))
        _eqnotmissing(x, upper) ? zero(result) : result
    elseif _eqnotmissing(x, upper)
        _logccdf_inclusive(d0, S(x))
    else
        result = logpdf(d0, S(x))
        _in_open_interval(x, lower, upper) ? result : oftype(result, -Inf)
    end
end

function loglikelihood(d::Censored{<:Any,<:Any,T}, x::AbstractArray{<:Real}) where {T}
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    S = float(Base.promote_eltype(T, first(x)))
    log_prob_lower = lower === missing ? 0 : logcdf(d0, S(lower))
    log_prob_upper = upper === missing ? 0 : _logccdf_inclusive(d0, S(upper))
    logzero = S(-Inf)

    return sum(x) do xi
        R = float(Base.promote_eltype(T, xi))
        _in_open_interval(xi, lower, upper) && return logpdf(d0, R(xi))
        _eqnotmissing(xi, lower) && return log_prob_lower
        _eqnotmissing(xi, upper) && return log_prob_upper
        return logzero
    end
end

function cdf(d::Censored{<:Any,<:Any,T}, x::Real) where {T}
    lower = d.lower
    upper = d.upper
    S = Base.promote_eltype(T, x)
    result = cdf(d.uncensored, S(x))
    return if lower !== missing && x < lower
        zero(result)
    elseif upper === missing || x < upper
        result
    else
        one(result)
    end
end

function logcdf(d::Censored{<:Any,<:Any,T}, x::Real) where {T}
    lower = d.lower
    upper = d.upper
    S = float(Base.promote_eltype(T, x))
    result = logcdf(d.uncensored, S(x))
    return if d.lower !== missing && x < d.lower
        oftype(result, -Inf)
    elseif d.upper === missing || x < d.upper
        result
    else
        zero(result)
    end
end

function ccdf(d::Censored{<:Any,<:Any,T}, x::Real) where {T}
    lower = d.lower
    upper = d.upper
    S = Base.promote_eltype(T, x)
    result = ccdf(d.uncensored, S(x))
    return if lower !== missing && x < lower
        one(result)
    elseif upper === missing || x < upper
        result
    else
        zero(result)
    end
end

function logccdf(d::Censored{<:Any,<:Any,T}, x::Real) where {T}
    lower = d.lower
    upper = d.upper
    S = float(Base.promote_eltype(T, x))
    result = logccdf(d.uncensored, S(x))
    return if lower !== missing && x < lower
        zero(result)
    elseif upper === missing || x < upper
        result
    else
        oftype(result, -Inf)
    end
end


#### Sampling

rand(rng::AbstractRNG, d::Censored) = _clamp(rand(rng, d.uncensored), d.lower, d.upper)


#### Utilities

# utilities to handle intervals represented with possibly missing bounds

_in_open_interval(x::Real, l::Real, u::Real) = l < x < u
_in_open_interval(x::Real, ::Missing, u::Real) = x < u
_in_open_interval(x::Real, l::Real, ::Missing) = x > l

function _to_truncated(d::Censored)
    return truncated(
        d.uncensored,
        d.lower === missing ? -Inf : d.lower,
        d.upper === missing ? Inf : d.upper,
    )
end

_clamp(x, l, u) = clamp(x, l, u)
_clamp(x, ::Missing, u) = min(x, u)
_clamp(x, l, ::Missing) = max(x, l)

_eqnotmissing(x::Real, y::Real) = x == y
_eqnotmissing(::Real, ::Missing) = false

# utilities for non-inclusive CDF p(x < u) and inclusive CCDF (p ≥ u)
_logcdf_noninclusive(d::UnivariateDistribution, x) = logcdf(d, x)
function _logcdf_noninclusive(d::DiscreteUnivariateDistribution, x)
    return logsubexp(logcdf(d, x), logpdf(d, x))
end

_ccdf_inclusive(d::UnivariateDistribution, x) = ccdf(d, x)
_ccdf_inclusive(d::DiscreteUnivariateDistribution, x) = ccdf(d, x) + pdf(d, x)

_logccdf_inclusive(d::UnivariateDistribution, x) = logccdf(d, x)
function _logccdf_inclusive(d::DiscreteUnivariateDistribution, x)
    return logaddexp(logccdf(d, x), logpdf(d, x))
end

# like xlogx but for input on log scale, safe when x == -Inf
function xexpx(x::Real)
    result = x * exp(x)
    return x == -Inf ? zero(result) : result
end
