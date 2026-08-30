"""
    Alpha(α, β)

The *Alpha distribution* with shape parameter `α` and scale parameter `β` has probability density function

```math
f(x; \\alpha, \\beta) = \\frac{\\beta}{\\sqrt{2\\pi} x^2 \\Phi(\\alpha)}
\\exp{\\bigg(-\\frac{(\\alpha-\\beta/x)^2}{2}\\bigg)}
```
where ``\\Phi`` is the normal CDF.


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
* [Reliability Application of the Alpha Distribution](https://ieeexplore.ieee.org/abstract/document/5222136)

"""
struct Alpha{T<:Real} <: ContinuousUnivariateDistribution
    α::T
    β::T
    Alpha{T}(α::T, β::T) where {T} = new{T}(α, β)
end

# as per NIST. ALPPDF. Online: https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/alppdf.htm
function Alpha(α::Real; check_args::Bool=true)
    @check_args Alpha (α, α > zero(α))
    return Alpha{typeof(α)}(α, oneunit(α))
end

function Alpha(α::Real, β::Real; check_args::Bool=true)
    @check_args Alpha (α, α > zero(α)) (β, β > zero(β))
    a, b = promote(α, β)
    return Alpha{typeof(a)}(a, b)
end

Alpha() = Alpha{Float64}(1.0, 1.0)

@distr_support Alpha 0.0 Inf

#### Conversions
Base.convert(::Type{Alpha{T}}, d::Alpha) where {T<:Real} = Alpha{T}(T(d.α), T(d.β))
Base.convert(::Type{Alpha{T}}, d::Alpha{T}) where {T<:Real} = d

#### Parameters

shape(d::Alpha) = d.α
scale(d::Alpha) = d.β

params(d::Alpha) = (d.α, d.β)
partype(::Alpha{T}) where {T} = T

#### Statistics

mode(d::Alpha) = ((sqrt(d.α^2 + 8) - d.α) * d.β) / 4

function pdf(d::Alpha, x::Real)
    res = d.β * normpdf(d.α - d.β/x) / (x^2 * normcdf(d.α))
    return x < 0 ? zero(res) : res
end

function cdf(d::Alpha, x::Real)
    return normcdf(d.α - d.β/x) / normcdf(d.α)
end

function logpdf(d::Alpha, x::Real)
    res = log(d.β / x^2) + normlogpdf(d.α - d.β/x) - normlogcdf(d.α)
    return x < 0 ? oftype(res, -Inf) : res
end

function logcdf(d::Alpha, x::Real)
    u = normlogcdf(d, d.α)
    v = normlogcdf(d, max(d.α - d.β/x, d.α))
    return u + log1mexp(v - u)
end

function quantile(d::Alpha, p::Real)
    return d.β / (d.α - norminvcdf(p * normcdf(d.α)))
end

function ccdf(d::Alpha, x::Real)
    return (normcdf(d.α) - normcdf(d.α - d.β/x)) / normcdf(d.α)
end