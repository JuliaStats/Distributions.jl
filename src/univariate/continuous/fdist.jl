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
immutable FDist <: ContinuousUnivariateDistribution
    ν1::Float64
    ν2::Float64

    function FDist(ν1::Real, ν2::Real)
        @check_args(FDist, ν1 > zero(ν1) && ν2 > zero(ν2))
        new(ν1, ν2)
    end
end

@distr_support FDist 0.0 Inf


#### Parameters

params(d::FDist) = (d.ν1, d.ν2)


#### Statistics

mean(d::FDist) = (ν2 = d.ν2; ν2 > 2.0 ? ν2 / (ν2 - 2.0) : NaN)

mode(d::FDist) = ((ν1, ν2) = params(d); ν1 > 2.0 ? ((ν1 - 2.0)/ν1) * (ν2 / (ν2 + 2.0)) : 0.0)

function var(d::FDist)
    (ν1, ν2) = params(d)
    ν2 > 4.0 ? 2.0 * ν2^2 * (ν1 + ν2 - 2.0) / (ν1 * (ν2 - 2.0)^2 * (ν2 - 4.0)) : NaN
end

function skewness(d::FDist)
    (ν1, ν2) = params(d)
    if ν2 > 6.0
        return (2.0 * ν1 + ν2 - 2.0) * sqrt(8.0 * (ν2 - 4.0)) / ((ν2 - 6.0) * sqrt(ν1 * (ν1 + ν2 - 2.0)))
    else
        return NaN
    end
end

function kurtosis(d::FDist)
    (ν1, ν2) = params(d)
    if ν2 > 8.0
        a = ν1 * (5. * ν2 - 22.) * (ν1 + ν2 - 2.) + (ν2 - 4.) * (ν2 - 2.)^2
        b = ν1 * (ν2 - 6.) * (ν2 - 8.) * (ν2 - 2.)
        return 12. * a / b
    else
        return NaN
    end
end

function entropy(d::FDist)
    (ν1, ν2) = params(d)
    hν1 = ν1 * 0.5
    hν2 = ν2 * 0.5
    hs = (ν1 + ν2) * 0.5
    return log(ν2 / ν1) + lgamma(hν1) + lgamma(hν2) - lgamma(hs) +
        (1.0 - hν1) * digamma(hν1) + (-1.0 - hν2) * digamma(hν2) +
        hs * digamma(hs)
end

#### Evaluation & Sampling

@_delegate_statsfuns FDist fdist ν1 ν2

rand(d::FDist) = StatsFuns.Rmath.fdistrand(d.ν1, d.ν2)
