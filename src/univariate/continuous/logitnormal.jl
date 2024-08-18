#using Optim # numerical estimation routines moved to NormalTransforms

"""
    LogitNormal(μ,σ)

The *logit normal distribution* is the distribution of
of a random variable whose logit has a [`Normal`](@ref) distribution.
Or inversely, when applying the logistic function to a Normal random variable
then the resulting random variable follows a logit normal distribution.

If ``X \\sim \\operatorname{Normal}(\\mu, \\sigma)`` then
``\\operatorname{logistic}(X) \\sim \\operatorname{LogitNormal}(\\mu,\\sigma)``.

The probability density function is

```math
f(x; \\mu, \\sigma) = \\frac{1}{x \\sqrt{2 \\pi \\sigma^2}}
\\exp \\left( - \\frac{(\\text{logit}(x) - \\mu)^2}{2 \\sigma^2} \\right),
\\quad x > 0
```

where the logit-Function is
```math
\\text{logit}(x) = \\ln\\left(\\frac{x}{1-x}\\right)
\\quad 0 < x < 1
```

```julia
LogitNormal()        # Logit-normal distribution with zero logit-mean and unit scale
LogitNormal(μ)       # Logit-normal distribution with logit-mean μ and unit scale
LogitNormal(μ, σ)    # Logit-normal distribution with logit-mean μ and scale σ

params(d)            # Get the parameters, i.e. (μ, σ)
median(d)            # Get the median, i.e. logistic(μ)
```

The following properties have no analytical solution but numerical
approximations. In order to avoid package dependencies for
numerical optimization, they are currently not implemented.

```julia
mean(d)
var(d)
std(d)
mode(d)
```

Similarly, skewness, kurtosis, and entropy are not implemented.

External links

* [Logit normal distribution on Wikipedia](https://en.wikipedia.org/wiki/Logit-normal_distribution)

"""
struct LogitNormal{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    LogitNormal{T}(μ::T, σ::T) where {T} = new{T}(μ, σ)
end

function LogitNormal(μ::T, σ::T; check_args::Bool=true) where {T <: Real}
    @check_args LogitNormal (σ, σ > zero(σ))
    return LogitNormal{T}(μ, σ)
end

LogitNormal(μ::Real, σ::Real; check_args::Bool=true) = LogitNormal(promote(μ, σ)...; check_args=check_args)
LogitNormal(μ::Integer, σ::Integer; check_args::Bool=true) = LogitNormal(float(μ), float(σ); check_args=check_args)
LogitNormal(μ::Real=0.0) = LogitNormal(μ, one(μ); check_args=false)

# minimum and maximum not defined for logitnormal
# but see https://github.com/JuliaStats/Distributions.jl/pull/457
@distr_support LogitNormal 0.0 1.0
#insupport(d::Union{D,Type{D}},x::Real) where {D<:LogitNormal} = 0.0 < x < 1.0


#### Conversions
convert(::Type{LogitNormal{T}}, μ::S, σ::S) where
  {T <: Real, S <: Real} = LogitNormal(T(μ), T(σ))
function Base.convert(::Type{LogitNormal{T}}, d::LogitNormal) where {T<:Real}
    LogitNormal{T}(T(d.μ), T(d.σ))
end
Base.convert(::Type{LogitNormal{T}}, d::LogitNormal{T}) where {T<:Real} = d

#### Parameters

params(d::LogitNormal) = (d.μ, d.σ)
location(d::LogitNormal) = d.μ
scale(d::LogitNormal) = d.σ
@inline partype(d::LogitNormal{T}) where {T<:Real} = T

#### Statistics

# meanlogitx(d::LogitNormal) = d.μ
# varlogitx(d::LogitNormal) = abs2(d.σ)
# stdlogitx(d::LogitNormal) = d.σ

# mean, mode, and variance
# moved to NormalTransforms because
# they depend on the Optim package.

# skewness, kurtosis, entropy
# not implemented

median(d::LogitNormal) = logistic(d.μ)

function kldivergence(p::LogitNormal, q::LogitNormal)
    pn = Normal{partype(p)}(p.μ, p.σ)
    qn = Normal{partype(q)}(q.μ, q.σ)
    return kldivergence(pn, qn)
end

#### Evaluation

#TODO check pd and logpdf
function pdf(d::LogitNormal{T}, x::Real) where T<:Real
    if zero(x) < x < one(x)
        return normpdf(d.μ, d.σ, logit(x)) / (x * (1-x))
    else
        return T(0)
    end
end

function logpdf(d::LogitNormal{T}, x::Real) where T<:Real
    if zero(x) < x < one(x)
        lx = logit(x)
        return normlogpdf(d.μ, d.σ, lx) - log(x) - log1p(-x)
    else
        return -T(Inf)
    end
end

cdf(d::LogitNormal{T}, x::Real) where {T<:Real} =
    x ≤ 0 ? zero(T) : x ≥ 1 ? one(T) : normcdf(d.μ, d.σ, logit(x))
ccdf(d::LogitNormal{T}, x::Real) where {T<:Real} =
    x ≤ 0 ? one(T) : x ≥ 1 ? zero(T) : normccdf(d.μ, d.σ, logit(x))
logcdf(d::LogitNormal{T}, x::Real) where {T<:Real} =
    x ≤ 0 ? -T(Inf) : x ≥ 1 ? zero(T) : normlogcdf(d.μ, d.σ, logit(x))
logccdf(d::LogitNormal{T}, x::Real) where {T<:Real} =
    x ≤ 0 ? zero(T) : x ≥ 1 ? -T(Inf) : normlogccdf(d.μ, d.σ, logit(x))

quantile(d::LogitNormal, q::Real) = logistic(norminvcdf(d.μ, d.σ, q))
cquantile(d::LogitNormal, q::Real) = logistic(norminvccdf(d.μ, d.σ, q))
invlogcdf(d::LogitNormal, lq::Real) = logistic(norminvlogcdf(d.μ, d.σ, lq))
invlogccdf(d::LogitNormal, lq::Real) = logistic(norminvlogccdf(d.μ, d.σ, lq))

function gradlogpdf(d::LogitNormal, x::Real)
    μ, σ = params(d)
    _insupport = insupport(d, x)
    _x = _insupport ? x : zero(x)
    z = (μ - logit(_x) + σ^2 * (2 * _x - 1)) / (σ^2 * _x * (1 - _x))
    return _insupport ? z : oftype(z, NaN)
end

# mgf(d::LogitNormal)
# cf(d::LogitNormal)


#### Sampling

xval(d::LogitNormal, z::Real) = logistic(muladd(d.σ, z, d.μ))

rand(rng::AbstractRNG, d::LogitNormal) = xval(d, randn(rng))
function rand!(rng::AbstractRNG, d::LogitNormal, A::AbstractArray{<:Real})
    randn!(rng, A)
    map!(Base.Fix1(xval, d), A, A)
    return A
end

## Fitting

function fit_mle(::Type{<:LogitNormal}, x::AbstractArray{T}) where T<:Real
    lx = logit.(x)
    μ, σ = mean_and_std(lx)
    LogitNormal(μ, σ)
end
