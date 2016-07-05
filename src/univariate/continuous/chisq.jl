doc"""
    Chisq(ν)
The *Chi squared distribution* (typically written χ²) with `ν` degrees of freedom has the
probability density function

$f(x; k) = \frac{x^{k/2 - 1} e^{-x/2}}{2^{k/2} \Gamma(k/2)}, \quad x > 0.$

If `ν` is an integer, then it is the distribution of the sum of squares of `ν` independent standard [`Normal`](:func:`Normal`) variates.

```julia
Chisq(k)     # Chi-squared distribution with k degrees of freedom

params(d)    # Get the parameters, i.e. (k,)
dof(d)       # Get the degrees of freedom, i.e. k
```

External links

* [Chi-squared distribution on Wikipedia](http://en.wikipedia.org/wiki/Chi-squared_distribution)
"""

immutable Chisq{T<:Real} <: ContinuousUnivariateDistribution
    ν::T

    Chisq(ν::T) = (@check_args(Chisq, ν > zero(ν)); new(ν))
end

Chisq{T<:Real}(ν::T) = Chisq{T}(ν)
Chisq(ν::Integer) = Chisq(Float64(ν))

@distr_support Chisq 0 Inf

#### Parameters

dof(d::Chisq) = d.ν
params(d::Chisq) = (d.ν,)

### Conversions
convert{T<:Real}(::Type{Chisq{T}}, ν::Real) = Chisq(T(ν))
convert{T <: Real, S <: Real}(::Type{Chisq{T}}, d::Chisq{S}) = Chisq(T(d.ν))


#### Statistics

mean(d::Chisq) = d.ν

var(d::Chisq) = 2d.ν

skewness(d::Chisq) = sqrt(8 / d.ν)

kurtosis(d::Chisq) = 12 / d.ν

mode{T<:Real}(d::Chisq{T}) = d.ν > 2 ? d.ν - 2 : zero(T)

function median(d::Chisq; approx::Bool=false)
    if approx
        return d.ν * (1 - 2 / (9 * d.ν))^3
    else
        return quantile(d, 1//2)
    end
end

function entropy(d::Chisq)
    hν = d.ν/2
    hν + logtwo + lgamma(hν) + (1 - hν) * digamma(hν)
end


#### Evaluation

@_delegate_statsfuns Chisq chisq ν

mgf(d::Chisq, t::Real) = (1 - 2 * t)^(-d.ν/2)

cf(d::Chisq, t::Real) = (1 - 2 * im * t)^(-d.ν/2)

gradlogpdf{T<:Real}(d::Chisq{T}, x::Real) =  x > 0 ? (d.ν/2 - 1) / x - 1//2 : zero(T)


#### Sampling

_chisq_rand(ν::Float64) = StatsFuns.RFunctions.chisqrand(ν)
rand(d::Chisq) = _chisq_rand(d.ν)
