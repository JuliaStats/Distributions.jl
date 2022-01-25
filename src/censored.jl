"""
    censored(d0::UnivariateDistribution; [lower::Real], [upper::Real])
    censored(d0::UnivariateDistribution, lower::Real, upper::Real)

A _censored distribution_ `d` of a distribution `d0` to the interval
``[l, u]=```[lower, upper]` has the probability density (mass) function:

```math
f(x; d_0, l, u) = \\begin{cases}
    P_{Z \\sim d_0}(Z \\le l), & x = l \\\\
    f_{d_0}(x),              & l < x < u \\\\
    P_{Z \\sim d_0}(Z \\ge u), & x = u \\\\
  \\end{cases}, \\quad x \\in [l, u]
```
where ``f_{d_0}(x)`` is the probability density (mass) function of ``d_0``.

If ``Z \\sim d_0``, and `X = clamp(Z, l, u)`, then ``X \\sim d``. Note that this implies
that even if ``d_0`` is continuous, its censored form assigns positive probability to the
bounds ``l`` and ``u``. Therefore, a censored continuous distribution has atoms and is a
mixture of discrete and continuous components.

```julia
censored(d0; lower=l)           # d0 left-censored to the interval [l, Inf)
censored(d0; upper=u)           # d0 right-censored to the interval (-Inf, u]
censored(d0; lower=l, upper=u)  # d0 interval-censored to the interval [l, u]
censored(d0, l, u)              # d0 interval-censored to the interval [l, u]
```

The function falls back to constructing a [`Distributions.Censored`](@ref) wrapper.

# Implementation

To implement a specialized censored form for distributions of type `D`, instead of
overloading a method with one of the above signatures, one or more of the following methods
should be implemented:
- `censored(d0::D, l::T, u::T) where {T <: Real}`
- `censored(d0::D, ::Nothing, u::Real)`
- `censored(d0::D, l::Real, ::Nothing)`
"""
function censored end
function censored(d0::UnivariateDistribution, l::T, u::T) where {T<:Real}
    return Censored(d0, l, u)
end
function censored(d0::UnivariateDistribution, ::Nothing, u::Real)
    return Censored(d0, nothing, u)
end
function censored(d0::UnivariateDistribution, l::Real, ::Nothing)
    return Censored(d0, l, nothing)
end
censored(d0::UnivariateDistribution, l::Real, u::Real) = censored(d0, promote(l, u)...)
censored(d0::UnivariateDistribution, ::Nothing, ::Nothing) = d0
function censored(
    d0::UnivariateDistribution;
    lower::Union{Real,Nothing} = nothing,
    upper::Union{Real,Nothing} = nothing,
)
    return censored(d0, lower, upper)
end

"""
    Censored

Generic wrapper for a [`censored`](@ref) distribution.
"""
struct Censored{
    D<:UnivariateDistribution,
    S<:ValueSupport,
    T<:Real,
    TL<:Union{T,Nothing},
    TU<:Union{T,Nothing},
} <: UnivariateDistribution{S}
    uncensored::D      # the original distribution (uncensored)
    lower::TL      # lower bound
    upper::TU      # upper bound
    function Censored(d0::UnivariateDistribution, lower::T, upper::T; check_args::Bool=true) where {T<:Real}
        check_args && @check_args(Censored, lower ≤ upper) 
        new{typeof(d0), value_support(typeof(d0)), T, T, T}(d0, lower, upper)
    end
    function Censored(d0::UnivariateDistribution, l::Nothing, u::Real; check_args::Bool=true)
        new{typeof(d0), value_support(typeof(d0)), typeof(u), Nothing, typeof(u)}(d0, l, u)
    end
    function Censored(d0::UnivariateDistribution, l::Real, u::Nothing; check_args::Bool=true)
        new{typeof(d0), value_support(typeof(d0)), typeof(l), typeof(l), Nothing}(d0, l, u)
    end
end

const RightCensored{D<:UnivariateDistribution,S<:ValueSupport,T<:Real} = Censored{D,S,T,Nothing,T}
const LeftCensored{D<:UnivariateDistribution,S<:ValueSupport,T<:Real} = Censored{D,S,T,T,Nothing}

function censored(d::Censored, l::T, u::T) where {T<:Real}
    return censored(
        d.uncensored,
        d.lower === nothing ? l : max(l, d.lower),
        d.upper === nothing ? u : min(u, d.upper),
    )
end
function censored(d::Censored, ::Nothing, u::Real)
    return censored(d.uncensored, d.lower, d.upper === nothing ? u : min(u, d.upper))
end
function censored(d::Censored, l::Real, ::Nothing)
    return censored(d.uncensored, d.lower === nothing ? l : max(l, d.lower), d.upper)
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

islowerbounded(d::RightCensored) = islowerbounded(d.uncensored)
islowerbounded(d::Censored) = islowerbounded(d.uncensored) || cdf(d.uncensored, d.lower) > 0

isupperbounded(d::LeftCensored) = isupperbounded(d.uncensored)
isupperbounded(d::Censored) = isupperbounded(d.uncensored) || _ccdf_inclusive(d.uncensored, d.upper) > 0

minimum(d::RightCensored) = min(minimum(d.uncensored), d.upper)
minimum(d::Censored) = max(minimum(d.uncensored), d.lower)

maximum(d::LeftCensored) = max(maximum(d.uncensored), d.lower)
maximum(d::Censored) = min(maximum(d.uncensored), d.upper)

function insupport(d::Censored, x::Real)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    return (
        (_in_open_interval(x, lower, upper) && insupport(d0, x)) ||
        (x == lower && cdf(d0, lower) > 0) ||
        (x == upper && _ccdf_inclusive(d0, upper) > 0)
    )
end

#### Show

function show(io::IO, ::MIME"text/plain", d::Censored)
    print(io, "Censored(")
    d0 = d.uncensored
    uml, namevals = _use_multline_show(d0)
    uml ? show_multline(io, d0, namevals; newline=false) : show_oneline(io, d0, namevals)
    if d.lower === nothing
        print(io, "; upper=$(d.upper))")
    elseif d.upper === nothing
        print(io, "; lower=$(d.lower))")
    else
        print(io, "; lower=$(d.lower), upper=$(d.upper))")
    end
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

function mean(d::RightCensored)
    upper = d.upper
    log_prob_upper = logccdf(d.uncensored, upper)
    prob_upper = exp(log_prob_upper)
    μ = prob_upper * (iszero(prob_upper) ? oneunit(upper) : upper)
    prob_trunc = exp(log1mexp(log_prob_upper))
    return μ + oftype(μ, prob_trunc * mean(_to_truncated(d)))
end
function mean(d::LeftCensored)
    lower = d.lower
    log_prob_lower = _logcdf_noninclusive(d.uncensored, lower)
    prob_lower = exp(log_prob_lower)
    μ = prob_lower * (iszero(prob_lower) ? oneunit(lower) : lower)
    prob_trunc = exp(log1mexp(log_prob_lower))
    return μ + oftype(μ, prob_trunc * mean(_to_truncated(d)))
end
function mean(d::Censored)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    log_prob_lower = _logcdf_noninclusive(d0, lower)
    prob_lower = exp(log_prob_lower)
    log_prob_upper = logccdf(d0, upper)
    prob_upper = exp(log_prob_upper)
    prob_trunc = exp(log1mexp(logaddexp(log_prob_lower, log_prob_upper)))
    μ = prob_trunc * mean(_to_truncated(d))
    if !iszero(prob_lower)
        μ += prob_lower * lower
    end
    if !iszero(prob_upper)
        μ += prob_upper * upper
    end
    return μ
end

function var(d::RightCensored)
    upper = d.upper
    log_prob_upper = logccdf(d.uncensored, upper)
    prob_upper = exp(log_prob_upper)
    μ_upper = prob_upper * (iszero(prob_upper) ? oneunit(upper) : upper)
    prob_trunc = exp(log1mexp(log_prob_upper))
    dtrunc = _to_truncated(d)
    μ_trunc = mean(dtrunc)
    μ = prob_trunc * μ_trunc + μ_upper
    v = prob_trunc * (var(dtrunc) + abs2(μ_trunc - μ))
    if !iszero(prob_upper)
        v += prob_upper * abs2(upper - μ)
    end
    return v
end
function var(d::LeftCensored)
    lower = d.lower
    log_prob_lower = _logcdf_noninclusive(d.uncensored, lower)
    prob_lower = exp(log_prob_lower)
    μ_lower = prob_lower * (iszero(prob_lower) ? oneunit(lower) : lower)
    prob_trunc = exp(log1mexp(log_prob_lower))
    dtrunc = _to_truncated(d)
    μ_trunc = mean(dtrunc)
    μ = prob_trunc * μ_trunc + μ_lower
    v = prob_trunc * (var(dtrunc) + abs2(μ_trunc - μ))
    if !iszero(prob_lower)
        v += prob_lower * abs2(lower - μ)
    end
    return v
end
function var(d::Censored)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    log_prob_lower = _logcdf_noninclusive(d0, lower)
    log_prob_upper = logccdf(d0, upper)
    prob_lower = exp(log_prob_lower)
    prob_upper = exp(log_prob_upper)
    prob_trunc = exp(log1mexp(logaddexp(log_prob_lower, log_prob_upper)))
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

function entropy(d::RightCensored)
    d0 = d.uncensored
    upper = d.upper
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
    prob_trunc = exp(log_prob_trunc)
    dtrunc = _to_truncated(d)
    entropy_interval = prob_trunc * entropy(dtrunc) - xexpx(log_prob_trunc) + xlogx_pu
    return oftype(entropy_bound, entropy_bound + entropy_interval)
end
function entropy(d::LeftCensored)
    d0 = d.uncensored
    lower = d.lower
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
    prob_trunc = exp(log_prob_trunc)
    dtrunc = _to_truncated(d)
    entropy_interval = prob_trunc * entropy(dtrunc) - xexpx(log_prob_trunc) + xlogx_pl
    return oftype(entropy_bound, entropy_bound + entropy_interval)
end
function entropy(d::Censored)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
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
    prob_trunc = exp(log_prob_trunc)
    dtrunc = _to_truncated(d)
    entropy_interval = prob_trunc * entropy(dtrunc) - xexpx(log_prob_trunc) + xlogx_pl + xlogx_pu
    return entropy_bound + entropy_interval
end


#### Evaluation

function pdf(d::Censored, x::Real)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    px = float(pdf(d0, x))
    return if _in_open_interval(x, lower, upper)
        px
    elseif x == lower
        x == upper ? one(px) : oftype(px, cdf(d0, x))
    elseif x == upper
        if value_support(typeof(d0)) === Discrete
            oftype(px, ccdf(d0, x) + px)
        else
            oftype(px, ccdf(d0, x))
        end
    else  # not in support
        zero(px)
    end
end

function logpdf(d::Censored, x::Real)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    logpx = logpdf(d0, x)
    return if _in_open_interval(x, lower, upper)
        logpx
    elseif x == lower
        x == upper ? zero(logpx) : oftype(logpx, logcdf(d0, x))
    elseif x == upper
        if value_support(typeof(d0)) === Discrete
            oftype(logpx, logaddexp(logccdf(d0, x), logpx))
        else
            oftype(logpx, logccdf(d0, x))
        end
    else  # not in support
        oftype(logpx, -Inf)
    end
end

function loglikelihood(d::Censored, x::AbstractArray{<:Real})
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    logpx = logpdf(d0, first(x))
    log_prob_lower = lower === nothing ? zero(logpx) : oftype(logpx, logcdf(d0, lower))
    log_prob_upper = upper === nothing ? zero(logpx) : oftype(logpx, _logccdf_inclusive(d0, upper))
    logzero = oftype(logpx, -Inf)
    return sum(x) do xi
        _in_open_interval(xi, lower, upper) && return logpdf(d0, xi)
        xi == lower && return log_prob_lower
        xi == upper && return log_prob_upper
        return logzero
    end
end

function cdf(d::Censored, x::Real)
    lower = d.lower
    upper = d.upper
    result = cdf(d.uncensored, x)
    return if lower !== nothing && x < lower
        zero(result)
    elseif upper === nothing || x < upper
        result
    else
        one(result)
    end
end

function logcdf(d::Censored, x::Real)
    lower = d.lower
    upper = d.upper
    result = logcdf(d.uncensored, x)
    return if d.lower !== nothing && x < d.lower
        oftype(result, -Inf)
    elseif d.upper === nothing || x < d.upper
        result
    else
        zero(result)
    end
end

function ccdf(d::Censored, x::Real)
    lower = d.lower
    upper = d.upper
    result = ccdf(d.uncensored, x)
    return if lower !== nothing && x < lower
        one(result)
    elseif upper === nothing || x < upper
        result
    else
        zero(result)
    end
end

function logccdf(d::Censored{<:Any,<:Any,T}, x::Real) where {T}
    lower = d.lower
    upper = d.upper
    result = logccdf(d.uncensored, x)
    return if lower !== nothing && x < lower
        zero(result)
    elseif upper === nothing || x < upper
        result
    else
        oftype(result, -Inf)
    end
end


#### Sampling

rand(rng::AbstractRNG, d::Censored) = _clamp(rand(rng, d.uncensored), d.lower, d.upper)


#### Utilities

# utilities to handle intervals represented with possibly `nothing` bounds

_in_open_interval(x::Real, l::Real, u::Real) = l < x < u
_in_open_interval(x::Real, ::Nothing, u::Real) = x < u
_in_open_interval(x::Real, l::Real, ::Nothing) = x > l

_clamp(x, l, u) = clamp(x, l, u)
_clamp(x, ::Nothing, u) = min(x, u)
_clamp(x, l, ::Nothing) = max(x, l)

function _to_truncated(d::Censored{<:UnivariateDistribution,<:ValueSupport,T}) where {T}
    FT = float(T)
    return truncated(
        d.uncensored,
        d.lower === nothing ? FT(-Inf) : d.lower,
        d.upper === nothing ? FT(Inf) : d.upper,
    )
end

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
