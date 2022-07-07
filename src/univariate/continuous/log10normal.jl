"""
    Log10Normal(μ,σ)

The *log10 normal distribution* is the distribution of the base 10 exponential of a [`Normal`](@ref) variate: if ``X \\sim \\operatorname{Normal}(\\mu, \\sigma)`` then
``\\exp10(X) \\sim \\operatorname{Log10Normal}(\\mu,\\sigma)``. The probability density function is
```math
f(x; \\mu, \\sigma) = \\frac{1}{x \\ln10 \\sqrt{2 \\pi \\sigma^2}}
\\exp \\left( - \\frac{(\\log10(x) - \\mu)^2}{2 \\sigma^2} \\right),
\\quad x > 0
```
```julia
Log10Normal()          # Log10-normal distribution with zero log10-mean and unit scale
Log10Normal(μ)         # Log10-normal distribution with log10-mean `μ` and unit scale
Log10Normal(μ, σ)      # Log10-normal distribution with log10-mean `μ` and scale `σ`

params(d)              # Get the parameters, i.e. (μ, σ)
```
External links

* [Log normal distribution on Wikipedia](http://en.wikipedia.org/wiki/Log-normal_distribution)

"""
struct Log10Normal{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    Log10Normal{T}(μ::T, σ::T) where {T} = new{T}(μ, σ)
end

function Log10Normal(μ::T, σ::T; check_args::Bool=true) where {T <: Real}
    @check_args Log10Normal (σ, σ ≥ zero(σ))
    return Log10Normal{T}(μ, σ)
end

Log10Normal(μ::Real, σ::Real; check_args::Bool=true) = Log10Normal(promote(μ, σ)...; check_args=check_args)
Log10Normal(μ::Integer, σ::Integer; check_args::Bool=true) = Log10Normal(float(μ), float(σ); check_args=check_args)
Log10Normal(μ::Real=0.0) = Log10Normal(μ, one(μ); check_args=false)

@distr_support Log10Normal 0.0 Inf

#### Conversions
convert(::Type{Log10Normal{T}}, μ::S, σ::S) where {T <: Real, S <: Real} = Log10Normal(T(μ), T(σ))
Base.convert(::Type{Log10Normal{T}}, d::Log10Normal) where {T<:Real} = Log10Normal{T}(T(d.μ), T(d.σ))
Base.convert(::Type{Log10Normal{T}}, d::Log10Normal{T}) where {T<:Real} = d

#### Parameters

params(d::Log10Normal) = (d.μ, d.σ)
partype(::Log10Normal{T}) where {T} = T

#### Statistics

mean(d::Log10Normal) = ((μ, σ) = params(d); exp10(μ + σ^2*log(10)/2))
median(d::Log10Normal) = exp10(d.μ)
mode(d::Log10Normal) = ((μ, σ) = params(d); exp10(μ - σ^2*log(10)))

function var(d::Log10Normal)
    σ2 = d.σ^2
    σ2 *= log(one(σ2)*10)^2
    e = exp(σ2)
    μ = d.μ
    10^2μ * e * (e-1)
end
# function vartest(d::Log10Normal)
#     quadgk(x-> (exp10(x)-mean(d))^2 * pdf(d,exp10(x)) * exp10(x) * log(10), min(1e-5,d.μ-5*d.σ),d.μ+5*d.σ)[1]
# end
# function meantest(d::Log10Normal)
#     quadgk(x-> exp10(x) * pdf(d,exp10(x)) * exp10(x) * log(10), min(1e-5,d.μ-5*d.σ),d.μ+5*d.σ)[1]
# end

function skewness(d::Log10Normal)
    σ2 = d.σ^2
    σ2 *= log(one(σ2)*10)^2
    e = exp(σ2)
    (e + 2) * sqrt(e - 1)
end

function kurtosis(d::Log10Normal)
    σ2 = d.σ^2
    σ2 *= log(one(σ2)*10)^2
    e = exp(σ2)
    e2 = e * e
    e3 = e2 * e
    e4 = e3 * e
    e4 + 2*e3 + 3*e2 - 3
end

function entropy(d::Log10Normal)
    (μ, σ) = params(d)
    (1 + μ * log(100) + log(twoπ) + 2*log(σ*log(10)))/2
end
# function entropytest(d::Log10Normal)
#     quadgk(x-> -logpdf(d,exp10(x)) * pdf(d,exp10(x)) * exp10(x) * log(10), min(1e-5,d.μ-5*d.σ),d.μ+5*d.σ)
# end


#### Evalution

function pdf(d::Log10Normal, x::Real)
    if x ≤ zero(x)
        log10x = log10(zero(x))
        x = one(x)
    else
        log10x = log10(x)
    end
    return pdf(Normal(d.μ, d.σ), log10x) / x / log(one(x)*10)
end

function logpdf(d::Log10Normal, x::Real)
    if x ≤ zero(x)
        log10x = log10(zero(x))
        b = zero(log10x)
    else
        log10x = log10(x)
        b = log(x)
    end
    return logpdf(Normal(d.μ, d.σ), log10x) - b - log(log(one(x)*10))
end

function cdf(d::Log10Normal, x::Real)
    log10x = x ≤ zero(x) ? log10(zero(x)) : log10(x)
    return cdf(Normal(d.μ, d.σ), log10x)
end

function ccdf(d::Log10Normal, x::Real)
    log10x = x ≤ zero(x) ? log10(zero(x)) : log10(x)
    return ccdf(Normal(d.μ, d.σ), log10x)
end

function logcdf(d::Log10Normal, x::Real)
    log10x = x ≤ zero(x) ? log10(zero(x)) : log10(x)
    return logcdf(Normal(d.μ, d.σ), log10x)
end

function logccdf(d::Log10Normal, x::Real)
    log10x = x ≤ zero(x) ? log10(zero(x)) : log10(x)
    return logccdf(Normal(d.μ, d.σ), log10x)
end

quantile(d::Log10Normal, q::Real) = exp10(quantile(Normal(params(d)...), q))
cquantile(d::Log10Normal, q::Real) = exp10(cquantile(Normal(params(d)...), q))
invlogcdf(d::Log10Normal, lq::Real) = exp10(invlogcdf(Normal(params(d)...), lq))
invlogccdf(d::Log10Normal, lq::Real) = exp10(invlogccdf(Normal(params(d)...), lq))

function gradlogpdf(d::Log10Normal, x::Real)
    outofsupport = x ≤ zero(x)
    y = outofsupport ? one(x) : x
    μ,σ = params(d)
    ln10 = log(one(μ)*10)
    z = ( ln10 * (μ - σ^2 * ln10) - log(y) ) / (y * σ^2 * ln10^2)
    return outofsupport ? zero(z) : z
end

#### Sampling

rand(rng::AbstractRNG, d::Log10Normal) = exp10(randn(rng) * d.σ + d.μ)

## Fitting

function fit_mle(::Type{<:Log10Normal}, x::AbstractArray{T}) where T<:Real
    lx = log10.(x)
    μ, σ = mean_and_std(lx)
    Log10Normal(μ, σ)
end

function fit_mle(::Type{<:Log10Normal}, x::AbstractArray{T}, w::AbstractArray{S}) where {T<:Real,S<:Real}
    @assert size(x) == size(w)
    lx = log10.(x)
    μ = sum( lx .* w ) / sum(w)
    σ = sqrt( sum( (lx .- μ ).^2 .* w) / sum(w) )
    Log10Normal(μ, σ)
end
