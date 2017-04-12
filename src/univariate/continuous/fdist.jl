doc"""
    FDist(ν1,ν2)

The *F distribution* has probability density function

$f(x; \nu_1, \nu_2) = \frac{1}{x B(\nu_1/2, \nu_2/2)}
\sqrt{\frac{(\nu_1 x)^{\nu_1} \cdot \nu_2^{\nu_2}}{(\nu_1 x + \nu_2)^{\nu_1 + \nu_2}}},
\quad x>0$

It is related to the [`Chisq`](:func:`Chisq`) distribution via the property that if $X_1
\sim \operatorname{Chisq}(\nu_1)$ and $X_2 \sim \operatorname{Chisq}(\nu_2)$, then
$(X_1/\nu_1) / (X_2 / \nu_2) \sim FDist(\nu_1, \nu_2)`.


```julia
FDist(d1, d2)     # F-Distribution with parameters d1 and d2

params(d)         # Get the parameters, i.e. (d1, d2)
```

External links

* [F distribution on Wikipedia](http://en.wikipedia.org/wiki/F-distribution)
    """
immutable FDist{T<:Real} <: ContinuousUnivariateDistribution
    ν1::T
    ν2::T

    function (::Type{FDist{T}}){T}(ν1::T, ν2::T)
        @check_args(FDist, ν1 > zero(ν1) && ν2 > zero(ν2))
        new{T}(ν1, ν2)
    end
end

FDist{T<:Real}(ν1::T, ν2::T) = FDist{T}(ν1, ν2)
FDist(ν1::Integer, ν2::Integer) = FDist(Float64(ν1), Float64(ν2))
FDist(ν1::Real, ν2::Real) = FDist(promote(ν1, ν2)...)

@distr_support FDist 0.0 Inf

#### Conversions
function convert{T <: Real, S <: Real}(::Type{FDist{T}}, ν1::S, ν2::S)
    FDist(T(ν1), T(ν2))
end
function convert{T <: Real, S <: Real}(::Type{FDist{T}}, d::FDist{S})
    FDist(T(d.ν1), T(d.ν2))
end

#### Parameters

params(d::FDist) = (d.ν1, d.ν2)
@inline partype{T<:Real}(d::FDist{T}) = T


#### Statistics

mean{T<:Real}(d::FDist{T}) = (ν2 = d.ν2; ν2 > 2 ? ν2 / (ν2 - 2) : T(NaN))

function mode{T<:Real}(d::FDist{T})
    (ν1, ν2) = params(d)
    ν1 > 2 ? ((ν1 - 2)/ν1) * (ν2 / (ν2 + 2)) : zero(T)
end

function var{T<:Real}(d::FDist{T})
    (ν1, ν2) = params(d)
    ν2 > 4 ? 2ν2^2 * (ν1 + ν2 - 2) / (ν1 * (ν2 - 2)^2 * (ν2 - 4)) : T(NaN)
end

function skewness{T<:Real}(d::FDist{T})
    (ν1, ν2) = params(d)
    if ν2 > 6
        return (2ν1 + ν2 - 2) * sqrt(8(ν2 - 4)) / ((ν2 - 6) * sqrt(ν1 * (ν1 + ν2 - 2)))
    else
        return T(NaN)
    end
end

function kurtosis{T<:Real}(d::FDist{T})
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
    return log(ν2 / ν1) + lgamma(hν1) + lgamma(hν2) - lgamma(hs) +
        (1 - hν1) * digamma(hν1) + (-1 - hν2) * digamma(hν2) +
        hs * digamma(hs)
end

#### Evaluation & Sampling

@_delegate_statsfuns FDist fdist ν1 ν2

rand(d::FDist) = StatsFuns.RFunctions.fdistrand(d.ν1, d.ν2)
