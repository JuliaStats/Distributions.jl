"""
    Chi(ν)

The *Chi distribution* `ν` degrees of freedom has probability density function

```math
f(x; \\nu) = \\frac{1}{\\Gamma(\\nu/2)} 2^{1 - \\nu/2} x^{\\nu-1} e^{-x^2/2}, \\quad x > 0
```

It is the distribution of the square-root of a [`Chisq`](@ref) variate.

```julia
Chi(ν)       # Chi distribution with ν degrees of freedom

params(d)    # Get the parameters, i.e. (ν,)
dof(d)       # Get the degrees of freedom, i.e. ν
```

External links

* [Chi distribution on Wikipedia](http://en.wikipedia.org/wiki/Chi_distribution)

"""
struct Chi{T<:Real} <: ContinuousUnivariateDistribution
    ν::T
    Chi{T}(ν::T) where {T} = new{T}(ν)
end

function Chi(ν::Real; check_args::Bool=true)
    @check_args Chi (ν, ν > zero(ν))
    return Chi{typeof(ν)}(ν)
end

Chi(ν::Integer; check_args::Bool=true) = Chi(float(ν); check_args=check_args)

@distr_support Chi 0.0 Inf

### Conversions
convert(::Type{Chi{T}}, ν::Real) where {T<:Real} = Chi(T(ν))
Base.convert(::Type{Chi{T}}, d::Chi) where {T<:Real} = Chi{T}(T(d.ν))
Base.convert(::Type{Chi{T}}, d::Chi{T}) where {T<:Real} = d

#### Parameters

dof(d::Chi) = d.ν
params(d::Chi) = (d.ν,)
@inline partype(d::Chi{T}) where {T<:Real} = T


#### Statistics

mean(d::Chi) = (h = d.ν/2; sqrt2 * exp(loggamma(h + 1//2) - loggamma(h)))

var(d::Chi) = d.ν - mean(d)^2
_chi_skewness(μ::Real, σ::Real) = (σ2 = σ^2; σ3 = σ2 * σ; (μ / σ3) * (1 - 2σ2))

function skewness(d::Chi)
    μ = mean(d)
    σ = sqrt(d.ν - μ^2)
    _chi_skewness(μ, σ)
end

function kurtosis(d::Chi)
    μ = mean(d)
    σ = sqrt(d.ν - μ^2)
    γ = _chi_skewness(μ, σ)
    (2/σ^2) * (1 - μ * σ * γ - σ^2)
end

entropy(d::Chi{T}) where {T<:Real} = (ν = d.ν;
    loggamma(ν/2) - T(logtwo)/2 - ((ν - 1)/2) * digamma(ν/2) + ν/2)

function mode(d::Chi; check_args::Bool=true)
    ν = d.ν
    @check_args(
        Chi,
        (ν, ν >= 1, "Chi distribution has no mode when ν < 1"),
    )
    sqrt(ν - 1)
end

function kldivergence(p::Chi, q::Chi)
    pν = dof(p)
    qν = dof(q)
    pν2 = pν / 2
    return loggamma(qν / 2) - loggamma(pν2) + (pν - qν) * digamma(pν2) / 2
end


#### Evaluation

function logpdf(d::Chi, x::Real)
    ν, _x = promote(d.ν, x)
    xsq = _x^2
    val = (xlogy(ν - 1, xsq / 2) - xsq + logtwo) / 2 - loggamma(ν / 2)
    return x < zero(x) ? oftype(val, -Inf) : val
end

gradlogpdf(d::Chi{T}, x::Real) where {T<:Real} = x >= 0 ? (d.ν - 1) / x - x : zero(T)

for f in (:cdf, :ccdf, :logcdf, :logccdf)
    @eval $f(d::Chi, x::Real) = $f(Chisq(d.ν; check_args=false), max(x, 0)^2)
end

for f in (:quantile, :cquantile, :invlogcdf, :invlogccdf)
    @eval $f(d::Chi, p::Real) = sqrt($f(Chisq(d.ν; check_args=false), p))
end

#### Sampling

rand(rng::AbstractRNG, d::Chi) =
    (ν = d.ν; sqrt(rand(rng, Gamma(ν / 2.0, 2.0one(ν)))))

struct ChiSampler{S <: Sampleable{Univariate,Continuous}} <:
    Sampleable{Univariate,Continuous}
    s::S
end

rand(rng::AbstractRNG, s::ChiSampler) = sqrt(rand(rng, s.s))

sampler(d::Chi) = ChiSampler(sampler(Chisq(d.ν)))
