doc"""
    Rayleigh(σ)

The *Rayleigh distribution* with scale `σ` has probability density function

$f(x; \sigma) = \frac{x}{\sigma^2} e^{-\frac{x^2}{2 \sigma^2}}, \quad x > 0$

It is related to the [`Normal`](:func:`Normal`) distribution via the property that if $X, Y \sim \operatorname{Normal}(0,\sigma)$, independently, then $\sqrt{X^2 + Y^2} \sim \operatorname{Rayleigh}(\sigma)$.

```julia
Rayleigh()       # Rayleigh distribution with unit scale, i.e. Rayleigh(1)
Rayleigh(s)      # Rayleigh distribution with scale s

params(d)        # Get the parameters, i.e. (s,)
scale(d)         # Get the scale parameter, i.e. s
```

External links

* [Rayleigh distribution on Wikipedia](http://en.wikipedia.org/wiki/Rayleigh_distribution)

"""

immutable Rayleigh{T<:Real} <: ContinuousUnivariateDistribution
    σ::T

    Rayleigh(σ::T) = (@check_args(Rayleigh, σ > zero(σ)); new(σ))
end

Rayleigh{T<:Real}(σ::T) = Rayleigh{T}(σ)
Rayleigh{T <: Integer}(σ::T) = Rayleigh(Float64(σ))
Rayleigh() = Rayleigh(1.0)

@distr_support Rayleigh 0 Inf

#### Conversions

convert{T <: Real, S <: Real}(::Type{Rayleigh{T}}, σ::S) = Rayleigh(T(σ))
convert{T <: Real, S <: Real}(::Type{Rayleigh{T}}, d::Rayleigh{S}) = Rayleigh(T(d.σ))

#### Parameters

scale(d::Rayleigh) = d.σ
params(d::Rayleigh) = (d.σ,)


#### Statistics

mean(d::Rayleigh) = sqrthalfπ * d.σ
median(d::Rayleigh) = 1.177410022515474691 * d.σ   # sqrt(log(4)) = 1.177410022515474691
mode(d::Rayleigh) = d.σ

var(d::Rayleigh) = 0.429203673205103381 * d.σ^2   # (2 - π / 2) = 0.429203673205103381
std(d::Rayleigh) = 0.655136377562033553 * d.σ

skewness{T<:Real}(d::Rayleigh{T}) = 0.631110657818937138*one(T)
kurtosis{T<:Real}(d::Rayleigh{T}) = 0.245089300687638063*one(T)

entropy(d::Rayleigh) = 0.942034242170793776 + log(d.σ)


#### Evaluation

function pdf{T<:Real}(d::Rayleigh{T}, x::Real)
	σ2 = d.σ^2
	x > 0 ? (x / σ2) * exp(- (x^2) / (2σ2)) : zero(T)
end

function logpdf{T<:Real}(d::Rayleigh{T}, x::Real)
	σ2 = d.σ^2
	x > 0 ? log(x / σ2) - (x^2) / (2σ2) : -T(Inf)
end

logccdf(d::Rayleigh, x::Real) = - (x^2) / (2d.σ^2)
ccdf(d::Rayleigh, x::Real) = exp(logccdf(d, x))

cdf(d::Rayleigh, x::Real) = 1 - ccdf(d, x)
logcdf(d::Rayleigh, x::Real) = log1mexp(logccdf(d, x))

quantile(d::Rayleigh, p::Real) = sqrt(-2d.σ^2 * log1p(-p))


#### Sampling

rand(d::Rayleigh) = d.σ * sqrt(2 * randexp())
