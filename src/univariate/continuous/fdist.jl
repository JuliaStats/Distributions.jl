"""
    FDist(ν1, ν2)

The *F distribution* has probability density function

```math
f(x; \\nu_1, \\nu_2) = \\frac{1}{x B(\\nu_1/2, \\nu_2/2)}
\\sqrt{\\frac{(\\nu_1 x)^{\\nu_1} \\cdot \\nu_2^{\\nu_2}}{(\\nu_1 x + \\nu_2)^{\\nu_1 + \\nu_2}}}, \\quad x>0
```

It is related to the [`Chisq`](@ref) distribution via the property that if
``X_1 \\sim \\operatorname{Chisq}(\\nu_1)`` and ``X_2 \\sim \\operatorname{Chisq}(\\nu_2)``, then
``(X_1/\\nu_1) / (X_2 / \\nu_2) \\sim \\operatorname{FDist}(\\nu_1, \\nu_2)``.

```julia
FDist(ν1, ν2)     # F-Distribution with parameters ν1 and ν2

params(d)         # Get the parameters, i.e. (ν1, ν2)
```

External links

* [F distribution on Wikipedia](http://en.wikipedia.org/wiki/F-distribution)
"""
struct FDist{T<:Real} <: ContinuousUnivariateDistribution
    ν1::T
    ν2::T

    function FDist{T}(ν1::T, ν2::T; check_args::Bool=true) where T
        @check_args FDist (ν1, ν1 > zero(ν1)) (ν2, ν2 > zero(ν2))
        new{T}(ν1, ν2)
    end
end

FDist(ν1::T, ν2::T; check_args::Bool=true) where {T<:Real} = FDist{T}(ν1, ν2; check_args=check_args)
FDist(ν1::Integer, ν2::Integer; check_args::Bool=true) = FDist(float(ν1), float(ν2); check_args=check_args)
FDist(ν1::Real, ν2::Real; check_args::Bool=true) = FDist(promote(ν1, ν2)...; check_args=check_args)

@distr_support FDist 0.0 Inf

#### Conversions
function convert(::Type{FDist{T}}, ν1::S, ν2::S) where {T <: Real, S <: Real}
    FDist(T(ν1), T(ν2))
end
Base.convert(::Type{FDist{T}}, d::FDist) where {T<:Real} = FDist{T}(T(d.ν1), T(d.ν2))
Base.convert(::Type{FDist{T}}, d::FDist{T}) where {T<:Real} = d

#### Parameters

params(d::FDist) = (d.ν1, d.ν2)
@inline partype(d::FDist{T}) where {T<:Real} = T


#### Statistics

mean(d::FDist{T}) where {T<:Real} = (ν2 = d.ν2; ν2 > 2 ? ν2 / (ν2 - 2) : T(NaN))

function mode(d::FDist{T}) where T<:Real
    (ν1, ν2) = params(d)
    ν1 > 2 ? ((ν1 - 2)/ν1) * (ν2 / (ν2 + 2)) : zero(T)
end

function var(d::FDist{T}) where T<:Real
    (ν1, ν2) = params(d)
    ν2 > 4 ? 2ν2^2 * (ν1 + ν2 - 2) / (ν1 * (ν2 - 2)^2 * (ν2 - 4)) : T(NaN)
end

function skewness(d::FDist{T}) where T<:Real
    (ν1, ν2) = params(d)
    if ν2 > 6
        return (2ν1 + ν2 - 2) * sqrt(8(ν2 - 4)) / ((ν2 - 6) * sqrt(ν1 * (ν1 + ν2 - 2)))
    else
        return T(NaN)
    end
end

function kurtosis(d::FDist{T}) where T<:Real
    (ν1, ν2) = params(d)
    if ν2 > 8
        a = ν1 * (5ν2 - 22) * (ν1 + ν2 - 2) + (ν2 - 4) * (ν2 - 2)^2
        b = ν1 * (ν2 - 6) * (ν2 - 8) * (ν1 + ν2 - 2)
        return 12a / b
    else
        return T(NaN)
    end
end

function entropy(d::FDist)
    (ν1, ν2) = params(d)
    hν1 = ν1/2
    hν2 = ν2/2
    hs = (ν1 + ν2)/2
    return log(ν2 / ν1) + loggamma(hν1) + loggamma(hν2) - loggamma(hs) +
        (1 - hν1) * digamma(hν1) + (-1 - hν2) * digamma(hν2) +
        hs * digamma(hs)
end

#### Evaluation & Sampling

@_delegate_statsfuns FDist fdist ν1 ν2

rand(rng::AbstractRNG, d::FDist) =
    ((ν1, ν2) = params(d);
     (ν2 * rand(rng, Chisq(ν1))) / (ν1 * rand(rng, Chisq(ν2))))
