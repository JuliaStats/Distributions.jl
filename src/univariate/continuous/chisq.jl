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
immutable Chisq <: ContinuousUnivariateDistribution
    ν::Float64

    Chisq(ν::Real) = (@check_args(Chisq, ν > zero(ν)); new(ν))
end

@distr_support Chisq 0.0 Inf

#### Parameters

dof(d::Chisq) = d.ν
params(d::Chisq) = (d.ν,)


#### Statistics

mean(d::Chisq) = d.ν

var(d::Chisq) = 2.0 * d.ν

skewness(d::Chisq) = sqrt(8.0 / d.ν)

kurtosis(d::Chisq) = 12.0 / d.ν

mode(d::Chisq) = d.ν > 2.0 ? d.ν - 2.0 : 0.0

function median(d::Chisq; approx::Bool=false)
    if approx
        return d.ν * (1.0 - 2.0 / (9.0 * d.ν))^3
    else
        return quantile(d, 0.5)
    end
end

function entropy(d::Chisq)
    hν = 0.5 * d.ν
    hν + logtwo + lgamma(hν) + (1.0 - hν) * digamma(hν)
end


#### Evaluation

@_delegate_statsfuns Chisq chisq ν

mgf(d::Chisq, t::Real) = (1.0 - 2.0 * t)^(-d.ν * 0.5)

cf(d::Chisq, t::Real) = (1.0 - 2.0 * im * t)^(-d.ν * 0.5)

gradlogpdf(d::Chisq, x::Float64) =  x > 0.0 ? (d.ν * 0.5 - 1) / x - 0.5 : 0.0


#### Sampling

_chisq_rand(ν::Float64) = StatsFuns.Rmath.chisqrand(ν)
rand(d::Chisq) = _chisq_rand(d.ν)
