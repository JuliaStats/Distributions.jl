doc"""
    InverseGamma(α, θ)

The *inverse gamma distribution* with shape parameter `α` and scale `θ` has probability
density function

$f(x; \alpha, \theta) = \frac{\theta^\alpha x^{-(\alpha + 1)}}{\Gamma(\alpha)}
e^{-\frac{\theta}{x}}, \quad x > 0$

It is related to the [`Gamma`](:func:`Gamma`) distribution: if $X \sim \operatorname{Gamma}(\alpha, \beta)$, then $1 / X \sim \operatorname{InverseGamma}(\alpha, \beta^{-1})$.

.. code-block:: julia

    InverseGamma()        # Inverse Gamma distribution with unit shape and unit scale, i.e. InverseGamma(1, 1)
    InverseGamma(a)       # Inverse Gamma distribution with shape a and unit scale, i.e. InverseGamma(a, 1)
    InverseGamma(a, b)    # Inverse Gamma distribution with shape a and scale b

    params(d)        # Get the parameters, i.e. (a, b)
    shape(d)         # Get the shape parameter, i.e. a
    scale(d)         # Get the scale parameter, i.e. b

External links

* [Inverse gamma distribution on Wikipedia](http://en.wikipedia.org/wiki/Inverse-gamma_distribution)

"""
immutable InverseGamma{T<:Real} <: ContinuousUnivariateDistribution
    invd::Gamma{T}
    θ::T

    function (::Type{InverseGamma{T}}){T}(α, θ)
        @check_args(InverseGamma, α > zero(α) && θ > zero(θ))
        new{T}(Gamma(α, 1 / θ), θ)
    end
end

InverseGamma{T<:Real}(α::T, θ::T) = InverseGamma{T}(α, θ)
InverseGamma(α::Real, θ::Real) = InverseGamma(promote(α, θ)...)
InverseGamma(α::Integer, θ::Integer) = InverseGamma(Float64(α), Float64(θ))
InverseGamma(α::Real) = InverseGamma(α, 1.0)
InverseGamma() = InverseGamma(1.0, 1.0)

@distr_support InverseGamma 0.0 Inf

#### Conversions
convert{T <: Real, S <: Real}(::Type{InverseGamma{T}}, α::S, θ::S) = InverseGamma(T(α), T(θ))
convert{T <: Real, S <: Real}(::Type{InverseGamma{T}}, d::InverseGamma{S}) = InverseGamma(T(shape(d.invd)), T(d.θ))

#### Parameters

shape(d::InverseGamma) = shape(d.invd)
scale(d::InverseGamma) = d.θ
rate(d::InverseGamma) = scale(d.invd)

params(d::InverseGamma) = (shape(d), scale(d))
@inline partype{T<:Real}(d::InverseGamma{T}) = T


#### Parameters

mean{T<:Real}(d::InverseGamma{T}) = ((α, θ) = params(d); α  > 1 ? θ / (α - 1) : T(Inf))

mode(d::InverseGamma) = scale(d) / (shape(d) + 1)

function var{T<:Real}(d::InverseGamma{T})
    (α, θ) = params(d)
    α > 2 ? θ^2 / ((α - 1)^2 * (α - 2)) : T(Inf)
end

function skewness{T<:Real}(d::InverseGamma{T})
    α = shape(d)
    α > 3 ? 4sqrt(α - 2) / (α - 3) : T(NaN)
end

function kurtosis{T<:Real}(d::InverseGamma{T})
    α = shape(d)
    α > 4 ? (30α - 66) / ((α - 3) * (α - 4)) : T(NaN)
end

function entropy(d::InverseGamma)
    (α, θ) = params(d)
    α + lgamma(α) - (1 + α) * digamma(α) + log(θ)
end


#### Evaluation

pdf(d::InverseGamma, x::Real) = exp(logpdf(d, x))

function logpdf(d::InverseGamma, x::Real)
    (α, θ) = params(d)
    α * log(θ) - lgamma(α) - (α + 1) * log(x) - θ / x
end

cdf(d::InverseGamma, x::Real) = ccdf(d.invd, 1 / x)
ccdf(d::InverseGamma, x::Real) = cdf(d.invd, 1 / x)
logcdf(d::InverseGamma, x::Real) = logccdf(d.invd, 1 / x)
logccdf(d::InverseGamma, x::Real) = logcdf(d.invd, 1 / x)

quantile(d::InverseGamma, p::Real) = 1 / cquantile(d.invd, p)
cquantile(d::InverseGamma, p::Real) = 1 / quantile(d.invd, p)
invlogcdf(d::InverseGamma, p::Real) = 1 / invlogccdf(d.invd, p)
invlogccdf(d::InverseGamma, p::Real) = 1 / invlogcdf(d.invd, p)

function mgf{T<:Real}(d::InverseGamma{T}, t::Real)
    (a, b) = params(d)
    t == zero(t) ? one(T) : 2(-b*t)^(0.5a) / gamma(a) * besselk(a, sqrt(-4*b*t))
end

function cf{T<:Real}(d::InverseGamma{T}, t::Real)
    (a, b) = params(d)
    t == zero(t) ? one(T)+zero(T)*im : 2(-im*b*t)^(0.5a) / gamma(a) * besselk(a, sqrt(-4*im*b*t))
end


#### Evaluation

rand(d::InverseGamma) = 1 / rand(d.invd)

function _rand!(d::InverseGamma, A::AbstractArray)
    s = sampler(d.invd)
    for i = 1:length(A)
    	v = 1 / rand(s)
        @inbounds A[i] = v
    end
    A
end
