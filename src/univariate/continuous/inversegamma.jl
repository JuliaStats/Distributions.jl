immutable InverseGamma <: ContinuousUnivariateDistribution
    invd::Gamma
    θ::Float64

    function InverseGamma(α::Real, θ::Real)
        @check_args(InverseGamma, α > zero(α) && θ > zero(θ))
        new(Gamma(α, 1.0 / θ), θ)
    end

    InverseGamma(α::Real) = InverseGamma(α, 1.0)
    InverseGamma() = InverseGamma(1.0, 1.0)
end

@distr_support InverseGamma 0.0 Inf
@distr_boundaries InverseGamma :open :closed


#### Parameters

shape(d::InverseGamma) = shape(d.invd)
scale(d::InverseGamma) = d.θ
rate(d::InverseGamma) = scale(d.invd)

params(d::InverseGamma) = (shape(d), scale(d))


#### Parameters

mean(d::InverseGamma) = ((α, θ) = params(d); α  > 1.0 ? θ / (α - 1.0) : Inf)

mode(d::InverseGamma) = scale(d) / (shape(d) + 1.0)

function var(d::InverseGamma)
    (α, θ) = params(d)
    α > 2.0 ? θ^2 / ((α - 1.0)^2 * (α - 2.0)) : Inf
end

function skewness(d::InverseGamma)
    α = shape(d)
    α > 3.0 ? 4.0 * sqrt(α - 2.0) / (α - 3.0) : NaN
end

function kurtosis(d::InverseGamma)
    α = shape(d)
    α > 4.0 ? (30.0 * α - 66.0) / ((α - 3.0) * (α - 4.0)) : NaN
end

function entropy(d::InverseGamma)
    (α, θ) = params(d)
    α + lgamma(α) - (1.0 + α) * digamma(α) + log(θ)
end


#### Evaluation

pdf(d::InverseGamma, x::Float64) = exp(logpdf(d, x))

function logpdf(d::InverseGamma, x::Float64)
    if insupport(d, x)
        (α, θ) = params(d)
        return α * log(θ) - lgamma(α) - (α + 1.0) * log(x) - θ / x
    else
        return -Inf
    end
end

cdf(d::InverseGamma, x::Float64) = insupport(d,x)?ccdf(d.invd, 1.0 / x):0.0
ccdf(d::InverseGamma, x::Float64) = insupport(d,x)?cdf(d.invd, 1.0 / x):1.0
logcdf(d::InverseGamma, x::Float64) = insupport(d,x)?logccdf(d.invd, 1.0 / x):-Inf
logccdf(d::InverseGamma, x::Float64) = insupport(d,x)?logcdf(d.invd, 1.0 / x):0.0

quantile(d::InverseGamma, p::Float64) = 1.0 / cquantile(d.invd, p)
cquantile(d::InverseGamma, p::Float64) = 1.0 / quantile(d.invd, p)
invlogcdf(d::InverseGamma, p::Float64) = 1.0 / invlogccdf(d.invd, p)
invlogccdf(d::InverseGamma, p::Float64) = 1.0 / invlogcdf(d.invd, p)

function mgf(d::InverseGamma, t::Real)
    (a, b) = params(d)
    t == zero(t) ? 1.0 : 2.0*(-b*t)^(0.5a) / gamma(a) * besselk(a, sqrt(-4.0*b*t))
end

function cf(d::InverseGamma, t::Real)
    (a, b) = params(d)
    t == zero(t) ? 1.0+0.0im : 2.0*(-im*b*t)^(0.5a) / gamma(a) * besselk(a, sqrt(-4.0*im*b*t))
end


#### Evaluation

rand(d::InverseGamma) = 1.0 / rand(d.invd)

function _rand!(d::InverseGamma, A::AbstractArray)
    s = sampler(d.invd)
    for i = 1:length(A)
    	v = 1.0 / rand(s)
        @inbounds A[i] = v
    end
    A
end
