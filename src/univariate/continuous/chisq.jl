"""
    Chisq(ν)
The *Chi squared distribution* (typically written χ²) with `ν` degrees of freedom has the
probability density function

```math
f(x; \\nu) = \\frac{x^{\\nu/2 - 1} e^{-x/2}}{2^{\\nu/2} \\Gamma(\\nu/2)}, \\quad x > 0.
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
Base.convert(::Type{Chisq{T}}, d::Chisq) where {T<:Real} = Chisq{T}(T(d.ν))
Base.convert(::Type{Chisq{T}}, d::Chisq{T}) where {T<:Real} = d

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

function kldivergence(p::Chisq, q::Chisq)
    pν = dof(p)
    qν = dof(q)
    return kldivergence(Chi{typeof(pν)}(pν), Chi{typeof(qν)}(qν))
end



#### Evaluation

@_delegate_statsfuns Chisq chisq ν

mgf(d::Chisq, t::Real) = (1 - 2 * t)^(-d.ν/2)
function cgf(d::Chisq, t)
    ν = dof(d)
    return -ν/2 * log1p(-2*t)
end


cf(d::Chisq, t::Real) = (1 - 2 * im * t)^(-d.ν/2)

gradlogpdf(d::Chisq{T}, x::Real) where {T<:Real} =  x > 0 ? (d.ν/2 - 1) / x - 1//2 : zero(T)


#### Sampling

function rand(rng::AbstractRNG, d::Chisq)
    α = dof(d) / 2
    θ = oftype(α, 2)
    return rand(rng, Gamma{typeof(α)}(α, θ))
end

function sampler(d::Chisq)
    α = dof(d) / 2
    θ = oftype(α, 2)
    return sampler(Gamma{typeof(α)}(α, θ))
end
