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
\\exp \\left( - \\frac{(\\logit(x) - \\mu)^2}{2 \\sigma^2} \\right),
\\quad x > 0
```

where the logit-Function is
```math
logit(x) = \\ln(frac{x}{1-x})
\\quad 0 < x < 1
```

```julia
LogitNormal()          # Logit-normal distribution with zero logit-mean and unit scale
LogitNormal(mu)        # Logit-normal distribution with logit-mean mu and unit scale
LogitNormal(mu, sig)   # Logit-normal distribution with logit-mean mu and scale sig

params(d)            # Get the parameters, i.e. (mu, sig)
median(d)            # Get the median, i.e. logistic(mu)
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

    LogitNormal{T}(μ::T, σ::T) where {T} = (@check_args(LogitNormal, σ > zero(σ)); new{T}(μ, σ))
end

LogitNormal(μ::T, σ::T) where {T<:Real} = LogitNormal{T}(μ, σ)
LogitNormal(μ::Real, σ::Real) = LogitNormal(promote(μ, σ)...)
LogitNormal(μ::Integer, σ::Integer) = LogitNormal(Float64(μ), Float64(σ))
LogitNormal(μ::Real) = LogitNormal(μ, convert(typeof(μ),1.0))
LogitNormal() = LogitNormal(0.0, 1.0)

# minimum and maximum not defined for logitnormal
# but see https://github.com/JuliaStats/Distributions.jl/pull/457
@distr_support LogitNormal 0.0 1.0
#insupport(d::Union{D,Type{D}},x::Real) where {D<:LogitNormal} = 0.0 < x < 1.0


#### Conversions
convert(::Type{LogitNormal{T}}, μ::S, σ::S) where 
  {T <: Real, S <: Real} = LogitNormal(T(μ), T(σ))
convert(::Type{LogitNormal{T}}, d::LogitNormal{S}) where 
  {T <: Real, S <: Real} = LogitNormal(T(d.μ), T(d.σ))

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

#### Evalution

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
  0 < x < 1 ? normcdf(d.μ, d.σ, logit(x)) : zero(T)
ccdf(d::LogitNormal{T}, x::Real) where {T<:Real} = 
  0 < x < 1 ? normccdf(d.μ, d.σ, logit(x)) : one(T)
logcdf(d::LogitNormal{T}, x::Real) where {T<:Real} = 
  0 < x < 1 ? normlogcdf(d.μ, d.σ, logit(x)) : -T(Inf)
logccdf(d::LogitNormal{T}, x::Real) where {T<:Real} = 
  0 < x < 1 ? normlogccdf(d.μ, d.σ, logit(x)) : zero(T)

quantile(d::LogitNormal, q::Real) = logistic(norminvcdf(d.μ, d.σ, q))
cquantile(d::LogitNormal, q::Real) = logistic(norminvccdf(d.μ, d.σ, q))
invlogcdf(d::LogitNormal, lq::Real) = logistic(norminvlogcdf(d.μ, d.σ, lq))
invlogccdf(d::LogitNormal, lq::Real) = logistic(norminvlogccdf(d.μ, d.σ, lq))

function gradlogpdf(d::LogitNormal{T}, x::Real) where T<:Real
    #TODO check
    (μ, σ) = params(d)
    0 < x < 1 ? - ((log(x) - μ) / ((σ^2) + 1) * x * (1-x)) : zero(T)
end

# mgf(d::LogitNormal)
# cf(d::LogitNormal)


#### Sampling

rand(rng::AbstractRNG, d::LogitNormal) = logistic(randn(rng) * d.σ + d.μ)

## Fitting

function fit_mle(::Type{<:LogitNormal}, x::AbstractArray{T}) where T<:Real
    lx = logit.(x)
    μ, σ = mean_and_std(lx)
    LogitNormal(μ, σ)
end


