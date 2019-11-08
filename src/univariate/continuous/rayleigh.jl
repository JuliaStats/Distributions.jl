"""
    Rayleigh(σ)

The *Rayleigh distribution* with scale `σ` has probability density function

```math
f(x; \\sigma) = \\frac{x}{\\sigma^2} e^{-\\frac{x^2}{2 \\sigma^2}}, \\quad x > 0
```

It is related to the [`Normal`](@ref) distribution via the property that if ``X, Y \\sim \\operatorname{Normal}(0,\\sigma)``, independently, then
``\\sqrt{X^2 + Y^2} \\sim \\operatorname{Rayleigh}(\\sigma)``.

```julia
Rayleigh()       # Rayleigh distribution with unit scale, i.e. Rayleigh(1)
Rayleigh(s)      # Rayleigh distribution with scale s

params(d)        # Get the parameters, i.e. (s,)
scale(d)         # Get the scale parameter, i.e. s
```

External links

* [Rayleigh distribution on Wikipedia](http://en.wikipedia.org/wiki/Rayleigh_distribution)

"""
struct Rayleigh{T<:Real} <: ContinuousUnivariateDistribution
    σ::T
    Rayleigh{T}(σ::T) where {T<:Real} = new{T}(σ)
end

function Rayleigh(σ::T; check_args=true) where {T <: Real}
    check_args && @check_args(Rayleigh, σ > zero(σ))
    return Rayleigh{T}(σ)
end

Rayleigh(σ::Integer) = Rayleigh(float(σ))
Rayleigh() = Rayleigh(1.0, check_args=false)

@distr_support Rayleigh 0.0 Inf

#### Conversions

convert(::Type{Rayleigh{T}}, σ::S) where {T <: Real, S <: Real} = Rayleigh(T(σ))
convert(::Type{Rayleigh{T}}, d::Rayleigh{S}) where {T <: Real, S <: Real} = Rayleigh(T(d.σ), check_args=false)

#### Parameters

scale(d::Rayleigh) = d.σ
params(d::Rayleigh) = (d.σ,)
partype(::Rayleigh{T}) where {T<:Real} = T


#### Statistics

mean(d::Rayleigh) = sqrthalfπ * d.σ
median(d::Rayleigh{T}) where {T<:Real} = sqrt2 * sqrt(T(logtwo)) * d.σ # sqrt(log(4))
mode(d::Rayleigh) = d.σ

var(d::Rayleigh{T}) where {T<:Real} = (2 - T(π)/2) * d.σ^2
std(d::Rayleigh{T}) where {T<:Real} = sqrt(2 - T(π)/2) * d.σ

skewness(d::Rayleigh{T}) where {T<:Real} = 2 * sqrtπ * (T(π) - 3)/(4 - T(π))^(3/2)
kurtosis(d::Rayleigh{T}) where {T<:Real} = -(6*T(π)^2 - 24*T(π) +16)/(4 - T(π))^2

entropy(d::Rayleigh{T}) where {T<:Real} = 1 - T(logtwo)/2 + T(MathConstants.γ)/2 + log(d.σ)


#### Evaluation

function pdf(d::Rayleigh{T}, x::Real) where T<:Real
    σ2 = d.σ^2
    x > 0 ? (x / σ2) * exp(- (x^2) / (2σ2)) : zero(T)
end

function logpdf(d::Rayleigh{T}, x::Real) where T<:Real
    σ2 = d.σ^2
    x > 0 ? log(x / σ2) - (x^2) / (2σ2) : -T(Inf)
end

logccdf(d::Rayleigh{T}, x::Real) where {T<:Real} = x > 0 ? - (x^2) / (2d.σ^2) : zero(T)
ccdf(d::Rayleigh, x::Real) = exp(logccdf(d, x))

cdf(d::Rayleigh, x::Real) = 1 - ccdf(d, x)
logcdf(d::Rayleigh, x::Real) = log1mexp(logccdf(d, x))

quantile(d::Rayleigh, p::Real) = sqrt(-2d.σ^2 * log1p(-p))


#### Sampling

rand(rng::AbstractRNG, d::Rayleigh) = d.σ * sqrt(2 * randexp(rng))


#### Fitting

function fit_mle(::Type{<:Rayleigh}, x::AbstractArray{T}) where {T<:Real}
    # Compute MLE (and unbiasd estimator) of σ^2
    s2 = zero(T)
    for xi in x
        s2 += xi^2
    end

    s2 /= (2*length(x))
    return Rayleigh(sqrt(s2))
end
