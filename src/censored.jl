"""
    censored(d::UnivariateDistribution, l::Real, u::Real)

Censor a univariate distribution `d` to the interval `[l, u]`.

The lower bound `l` can be finite or `-Inf` and the upper bound `u` can be finite or
`Inf`. The function throws an error if `l > u`.

The function falls back to constructing a [`Censored`](@ref) wrapper.

# Implementation

To implement a specialized censored form for distributions of type `D`, the method
`censored(d::D, l::T, u::T) where {T <: Real}` should be implemented.
"""
function censored(d::UnivariateDistribution, l::Real, u::Real)
    return censored(d, promote(l, u)...)
end
censored(d::UnivariateDistribution, l::T, u::T) where {T <: Real} = Censored(d, l, u)

"""
    Censored

Generic wrapper for a censored distribution.
"""
struct Censored{D<:UnivariateDistribution, S<:ValueSupport, T <: Real} <: UnivariateDistribution{S}
    uncensored::D      # the original distribution (uncensored)
    lower::T      # lower bound
    upper::T      # upper bound
    function Censored(d::UnivariateDistribution, l::T, u::T) where {T <: Real}
        l ≤ u || error("the lower bound must be less than or equal to the upper bound")
        new{typeof(d), value_support(typeof(d)), T}(d, l, u)
    end
end

params(d::Censored) = tuple(params(d.uncensored)..., d.lower, d.upper)
partype(d::Censored) = partype(d.uncensored)
Base.eltype(::Type{<:Censored{D}}) where {D} = eltype(D)

#### Range and Support

islowerbounded(d::Censored) = islowerbounded(d.uncensored) || isfinite(d.lower)
isupperbounded(d::Censored) = isupperbounded(d.uncensored) || isfinite(d.upper)

minimum(d::Censored) = max(minimum(d.uncensored), d.lower)
maximum(d::Censored) = min(maximum(d.uncensored), d.upper)

function insupport(d::Censored{<:UnivariateDistribution}, x::Real)
    return d.lower ≤ x ≤ d.upper && insupport(d.uncensored, x)
end

#### Show

function show(io::IO, d::Censored)
    print(io, "Censored(")
    d0 = d.uncensored
    uml, namevals = _use_multline_show(d0)
    uml ? show_multline(io, d0, namevals) :
          show_oneline(io, d0, namevals)
    print(io, ", range=(", d.lower, ", ", d.upper, ")")
    uml && println(io)
end

_use_multline_show(d::Censored) = _use_multline_show(d.uncensored)


#### Statistics

quantile(d::Censored, p::Real) = clamp(quantile(d.uncensored, p), d.lower, d.upper)

median(d::Censored) = clamp(median(d.uncensored), d.lower, d.upper)

#### Evaluation

function pdf(d::Censored, x::Real)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    return if x == lower
        cdf(d0, x)
    elseif x == upper
        uccdf = ccdf(d0, x)
        if value_support(typeof(d)) === Discrete
            uccdf - pdf(d0, x)
        else
            uccdf
        end
    else
        result = pdf(d0, x)
        lower < x < upper ? result : zero(result)
    end
end

function logpdf(d::Censored, x::Real)
    d0 = d.uncensored
    lower = d.lower
    upper = d.upper
    return if x == lower
        logcdf(d0, x)
    elseif x == upper
        ulogccdf = logccdf(d0, x)
        if value_support(typeof(d)) === Discrete
            logsubexp(ulogccdf, logpdf(d0, x))
        else
            ulogccdf
        end
    else
        result = logpdf(d0, x)
        lower < x < upper ? result : oftype(result, -Inf)
    end
end

function cdf(d::Censored, x::Real)
    result = cdf(d.uncensored, x)
    return if x < d.lower
        zero(result)
    elseif x < d.upper
        result
    else
        one(result)
    end
end

function logcdf(d::Censored, x::Real)
    result = logcdf(d.uncensored, x)
    return if x < d.lower
        oftype(result, -Inf)
    elseif x < d.upper
        result
    else
        zero(result)
    end
end

function ccdf(d::Censored, x::Real)
    result = ccdf(d.uncensored, x)
    return if x > d.upper
        zero(result)
    elseif x > d.lower
        result
    else
        one(result)
    end
end

function logccdf(d::Censored, x::Real)
    result = logccdf(d.uncensored, x)
    return if x > d.upper
        oftype(result, -Inf)
    elseif x > d.lower
        result
    else
        zero(result)
    end
end


#### Sampling

function rand(rng::AbstractRNG, d::Censored)
    return clamp(rand(rng, d.uncensored), d.lower, d.upper)
end
