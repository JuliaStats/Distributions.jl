"""
    Alpha(α, β)

The *Alpha distribution* with shape parameter `α` and scale parameter `β` has probability density function

```math
f(x, \\alpha) = \\frac{\\beta}{x^2 \\Phi(\\alpha) \\sqrt{2\\pi}} *
\\exp(-\\frac{1}{2} (\\alpha-\\beta/x)^2)
```
where :math:`\\Phi` is the normal CDF.


```julia
Alpha()        # equivalent to Alpha distribution with unit shape and unit scale i.e. Alpha(1, 1)
Alpha(α)       # equivalent to Alpha distribution with shape α and unit scale, i.e. Alpha(α, 1)
Alpha(α, β)    # equivalent to Alpha distribution with shape α and scale β i.e. Alpha(α, β)

params(d)        # Get the parameters, i.e. (α, β)
shape(d)         # Get the shape parameter, i.e. α
scale(d)         # Get the scale parameter, i.e. β
```

External links

* [Continuous Univariate Distributions, Volumes I and II](https://www.wiley.com/en-us/Continuous+Univariate+Distributions%2C+Volume+1%2C+2nd+Edition-p-9780471584957)
* [scipy.stats.alpha](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.alpha.html)

"""
struct Alpha{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    β::T
    Alpha{T}(α, β) where {T} = new{T}(α, β)
end

# as per NIST. ALPPDF. Online: https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/alppdf.htm
function Alpha(α::T; check_args::Bool=true) where {T <: Real}
    @check_args Alpha (α, α > zero(α))
    return Alpha{T}(α, 1)
end

function Alpha(α::T, β::T; check_args::Bool=true) where {T <: Real}
    @check_args Alpha (α, α > zero(α)) (β, β > zero(β))
    return Alpha{T}(α, β)
end

Alpha(α::Real, β::Real; check_args::Bool=true) = Alpha(promote(α, β)...; check_args=check_args)
Alpha(α::Integer, β::Integer; check_args::Bool=true) = Alpha(float(α), float(β); check_args=check_args)
Alpha(α::Real; check_args::Bool=true) = Alpha(α, one(α); check_args=check_args)
Alpha() = Alpha{Float64}(1.0, 1.0)

@distr_support Alpha 0.0 Inf

#### Conversions
convert(::Type{Alpha{T}}, α::S, β::S) where {T <: Real, S <: Real} = Alpha(T(α), T(β))
Base.convert(::Type{Alpha{T}}, d::Alpha) where {T<:Real} = Alpha{T}(T(d.α), T(d.β))
Base.convert(::Type{Alpha{T}}, d::Alpha{T}) where {T<:Real} = d

#### Parameters

shape(d::Alpha) = d.α
scale(d::Alpha) = d.β

params(d::Alpha) = (d.α, d.β)
partype(::Alpha{T}) where {T} = T

#### Statistics

mode(d::Alpha) = ((sqrt(d.α^2 + 8) - d.α) * d.β) / 4

function pdf(d::Alpha{T}, x::Real) where T<:Real
    nd = Normal()
    return pdf(nd, d.α-d.β/x) / ((x^2) * cdf(nd, d.α))
end

function cdf(d::Alpha{T}, x::Real) where T<:Real
    nd = Normal()
    return cdf(nd, d.α-d.β/x) / cdf(nd, d.α)
end