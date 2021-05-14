"""
    LocationScale(μ,σ,ρ)

A location-scale transformed distribution with location parameter `μ`,
scale parameter `σ`, and given univariate distribution `ρ`.

If ``Z`` is a random variable with distribution `ρ`, then the distribution of the random
variable
```math
X = μ + σ Z
```
is the location-scale transformed distribution with location parameter `μ` and scale
parameter `σ`.

If `ρ` is a discrete distribution, the probability mass function of
the transformed distribution is given by
```math
P(X = x) = P\\left(Z = \\frac{x-μ}{σ} \\right).
```
If `ρ` is a continuous distribution, the probability density function of
the transformed distribution is given by
```math
f(x) = \\frac{1}{σ} ρ \\! \\left( \\frac{x-μ}{σ} \\right).
```

```julia
LocationScale(μ,σ,ρ) # location-scale transformed distribution
params(d)            # Get the parameters, i.e. (μ, σ, and the base distribution ρ)
location(d)          # Get the location parameter
scale(d)             # Get the scale parameter
```

External links
[Location-Scale family on Wikipedia](https://en.wikipedia.org/wiki/Location%E2%80%93scale_family)
"""
struct LocationScale{T<:Real, S<:ValueSupport, D<:UnivariateDistribution{S}} <: UnivariateDistribution{S}
    μ::T
    σ::T
    ρ::D
    function LocationScale{T,S,D}(μ::T, σ::T, ρ::D; check_args=true) where {T<:Real, S<:ValueSupport, D<:UnivariateDistribution{S}}
        check_args && @check_args(LocationScale, σ > zero(σ))
        new{T, S, D}(μ, σ, ρ)
    end
end

function LocationScale(μ::T, σ::T, ρ::UnivariateDistribution; check_args=true) where {T<:Real}
    _T = promote_type(eltype(ρ), T)
    D = typeof(ρ)
    S = value_support(D)
    return LocationScale{_T,S,D}(_T(μ), _T(σ), ρ; check_args=check_args)
end

LocationScale(μ::Real, σ::Real, ρ::UnivariateDistribution) = LocationScale(promote(μ, σ)..., ρ)

# aliases
const ContinuousLocationScale{T<:Real,D<:ContinuousUnivariateDistribution} = LocationScale{T,Continuous,D}
const DiscreteLocationScale{T<:Real,D<:DiscreteUnivariateDistribution} = LocationScale{T,Discrete,D}

Base.eltype(::Type{<:LocationScale{T}}) where T = T

minimum(d::LocationScale) = d.μ + d.σ * minimum(d.ρ)
maximum(d::LocationScale) = d.μ + d.σ * maximum(d.ρ)

LocationScale(μ::Real, σ::Real, d::LocationScale) = LocationScale(μ + d.μ * σ, σ * d.σ, d.ρ)

#### Conversions

convert(::Type{LocationScale{T}}, μ::Real, σ::Real, ρ::D) where {T<:Real, D<:UnivariateDistribution} = LocationScale(T(μ),T(σ),ρ)
convert(::Type{LocationScale{T}}, d::LocationScale{S}) where {T<:Real, S<:Real} = LocationScale(T(d.μ),T(d.σ),d.ρ, check_args=false)

#### Parameters

location(d::LocationScale) = d.μ
scale(d::LocationScale) = d.σ
params(d::LocationScale) = (d.μ,d.σ,d.ρ)
partype(::LocationScale{T}) where {T} = T

#### Statistics

mean(d::LocationScale) = d.μ + d.σ * mean(d.ρ)
median(d::LocationScale) = d.μ + d.σ * median(d.ρ)
mode(d::LocationScale) = d.μ + d.σ * mode(d.ρ)
modes(d::LocationScale) = d.μ .+ d.σ .* modes(d.ρ)

var(d::LocationScale) = d.σ^2 * var(d.ρ)
std(d::LocationScale) = d.σ * std(d.ρ)
skewness(d::LocationScale) = skewness(d.ρ)
kurtosis(d::LocationScale) = kurtosis(d.ρ)

isplatykurtic(d::LocationScale) = isplatykurtic(d.ρ)
isleptokurtic(d::LocationScale) = isleptokurtic(d.ρ)
ismesokurtic(d::LocationScale) = ismesokurtic(d.ρ)

entropy(d::ContinuousLocationScale) = entropy(d.ρ) + log(d.σ)
entropy(d::DiscreteLocationScale) = entropy(d.ρ)

mgf(d::LocationScale,t::Real) = exp(d.μ*t) * mgf(d.ρ,d.σ*t)

#### Evaluation & Sampling

pdf(d::ContinuousLocationScale,x::Real) = pdf(d.ρ,(x-d.μ)/d.σ) / d.σ
pdf(d::DiscreteLocationScale, x::Real) = pdf(d.ρ,(x-d.μ)/d.σ)

logpdf(d::ContinuousLocationScale,x::Real) = logpdf(d.ρ,(x-d.μ)/d.σ) - log(d.σ)
logpdf(d::DiscreteLocationScale, x::Real) = logpdf(d.ρ,(x-d.μ)/d.σ)

# additional definitions are required to fix ambiguity errors and incorrect defaults
for f in (:cdf, :ccdf, :logcdf, :logccdf)
    _f = Symbol(:_, f)
    @eval begin
        $f(d::LocationScale, x::Real) = $_f(d, x)
        $f(d::DiscreteLocationScale, x::Real) = $_f(d, x)
        $f(d::DiscreteLocationScale, x::Integer) = $_f(d, x)
        $_f(d::LocationScale, x::Real) = $f(d.ρ, (x - d.μ) / d.σ)
    end
end

quantile(d::LocationScale,q::Real) = d.μ + d.σ * quantile(d.ρ,q)

rand(rng::AbstractRNG, d::LocationScale) = d.μ + d.σ * rand(rng, d.ρ)
cf(d::LocationScale, t::Real) = cf(d.ρ,t*d.σ) * exp(1im*t*d.μ)
gradlogpdf(d::ContinuousLocationScale, x::Real) = gradlogpdf(d.ρ,(x-d.μ)/d.σ) / d.σ

#### Syntactic sugar for simple transforms of distributions, e.g., d + x, d - x, and so on

Base.:+(d::UnivariateDistribution, x::Real) = LocationScale(x, one(x), d)
Base.:+(x::Real, d::UnivariateDistribution) = d + x
Base.:*(x::Real, d::UnivariateDistribution) = LocationScale(zero(x), x, d)
Base.:*(d::UnivariateDistribution, x::Real) = x * d
Base.:-(d::UnivariateDistribution, x::Real) = d + -x
Base.:/(d::UnivariateDistribution, x::Real) = inv(x) * d

