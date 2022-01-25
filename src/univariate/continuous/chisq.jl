"""
    Chisq(ν)
The *Chi squared distribution* (typically written χ²) with `ν` degrees of freedom has the
probability density function

```math
f(x; \\nu) = \\frac{x^{\\nu/2 - 1} e^{-x/2}}{2^{\\nu/2} \\Gamma(k/2)}, \\quad x > 0.
```

If `ν` is an integer, then it is the distribution of the sum of squares of `ν` independent standard [`Normal`](@ref) variates.

```julia
Chisq(ν)     # Chi-squared distribution with ν degrees of freedom

params(d)    # Get the parameters, i.e. (ν,)
dof(d)       # Get the degrees of freedom, i.e. ν
```

External links

* [Chi-squared distribution on Wikipedia](http://en.wikipedia.org/wiki/Chi-squared_distribution)
"""
struct Chisq{T<:Real} <: ContinuousUnivariateDistribution
    ν::T
    Chisq{T}(ν::T) where {T} = new{T}(ν)
end

function Chisq(ν::Real; check_args::Bool=true)
    @check_args Chisq (ν, ν > zero(ν))
    return Chisq{typeof(ν)}(ν)
end

Chisq(ν::Integer; check_args::Bool=true) = Chisq(float(ν); check_args=check_args)

@distr_support Chisq 0.0 Inf

#### Parameters

dof(d::Chisq) = d.ν
params(d::Chisq) = (d.ν,)
@inline partype(d::Chisq{T}) where {T<:Real} = T

### Conversions
convert(::Type{Chisq{T}}, ν::Real) where {T<:Real} = Chisq(T(ν))
convert(::Type{Chisq{T}}, d::Chisq{S}) where {T <: Real, S <: Real} = Chisq(T(d.ν))


#### Statistics

mean(d::Chisq) = d.ν

var(d::Chisq) = 2d.ν

skewness(d::Chisq) = sqrt(8 / d.ν)

kurtosis(d::Chisq) = 12 / d.ν

mode(d::Chisq{T}) where {T<:Real} = d.ν > 2 ? d.ν - 2 : zero(T)

function median(d::Chisq; approx::Bool=false)
    if approx
        return d.ν * (1 - 2 / (9 * d.ν))^3
    else
        return quantile(d, 1//2)
    end
end

function entropy(d::Chisq)
    hν = d.ν/2
    hν + logtwo + loggamma(hν) + (1 - hν) * digamma(hν)
end


#### Evaluation

@_delegate_statsfuns Chisq chisq ν

mgf(d::Chisq, t::Real) = (1 - 2 * t)^(-d.ν/2)

cf(d::Chisq, t::Real) = (1 - 2 * im * t)^(-d.ν/2)

gradlogpdf(d::Chisq{T}, x::Real) where {T<:Real} =  x > 0 ? (d.ν/2 - 1) / x - 1//2 : zero(T)


#### Sampling

rand(rng::AbstractRNG, d::Chisq) =
    (ν = d.ν; rand(rng, Gamma(ν / 2.0, 2.0one(ν))))

sampler(d::Chisq) = (ν = d.ν; sampler(Gamma(ν / 2.0, 2.0one(ν))))
