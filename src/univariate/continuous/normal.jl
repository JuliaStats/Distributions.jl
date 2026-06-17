"""
    Normal(μ,σ)

The *Normal distribution* with mean `μ` and standard deviation `σ≥0` has probability density function

```math
f(x; \\mu, \\sigma) = \\frac{1}{\\sqrt{2 \\pi \\sigma^2}}
\\exp \\left( - \\frac{(x - \\mu)^2}{2 \\sigma^2} \\right)
```

Note that if `σ == 0`, then the distribution is a point mass concentrated at `μ`.
Though not technically a continuous distribution, it is allowed so as to account for cases where `σ` may have underflowed,
and the functions are defined by taking the pointwise limit as ``σ → 0``.

```julia
Normal()          # standard Normal distribution with zero mean and unit variance
Normal(μ)         # Normal distribution with mean μ and unit variance
Normal(μ, σ)      # Normal distribution with mean μ and variance σ^2

params(d)         # Get the parameters, i.e. (μ, σ)
mean(d)           # Get the mean, i.e. μ
std(d)            # Get the standard deviation, i.e. σ
```

External links

* [Normal distribution on Wikipedia](http://en.wikipedia.org/wiki/Normal_distribution)

"""
struct Normal{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T
    Normal{T}(µ::T, σ::T) where {T<:Real} = new{T}(µ, σ)
end

function Normal(μ::T, σ::T; check_args::Bool=true) where {T <: Real}
    @check_args Normal (σ, σ >= zero(σ))
    return Normal{T}(μ, σ)
end

#### Outer constructors
Normal(μ::Real, σ::Real; check_args::Bool=true) = Normal(promote(μ, σ)...; check_args=check_args)
Normal(μ::Integer, σ::Integer; check_args::Bool=true) = Normal(float(μ), float(σ); check_args=check_args)
Normal(μ::Real=0.0) = Normal(μ, one(μ); check_args=false)

const Gaussian = Normal

# #### Conversions
convert(::Type{Normal{T}}, μ::S, σ::S) where {T <: Real, S <: Real} = Normal(T(μ), T(σ))
Base.convert(::Type{Normal{T}}, d::Normal) where {T<:Real} = Normal{T}(T(d.μ), T(d.σ))
Base.convert(::Type{Normal{T}}, d::Normal{T}) where {T<:Real} = d

@distr_support Normal -Inf Inf

#### Parameters

params(d::Normal) = (d.μ, d.σ)
@inline partype(d::Normal{T}) where {T<:Real} = T

location(d::Normal) = d.μ
scale(d::Normal) = d.σ

Base.eltype(::Type{Normal{T}}) where {T} = T

#### Statistics

mean(d::Normal) = d.μ
median(d::Normal) = d.μ
mode(d::Normal) = d.μ

var(d::Normal) = abs2(d.σ)
std(d::Normal) = d.σ
skewness(d::Normal{T}) where {T<:Real} = zero(T)
kurtosis(d::Normal{T}) where {T<:Real} = zero(T)

entropy(d::Normal) = (log2π + 1)/2 + log(d.σ)

function kldivergence(p::Normal, q::Normal)
    μp = mean(p)
    σ²p = var(p)
    μq = mean(q)
    σ²q = var(q)
    σ²p_over_σ²q = σ²p / σ²q
    return (abs2(μp - μq) / σ²q - logmxp1(σ²p_over_σ²q)) / 2
end

#### Evaluation

# Use Julia implementations in StatsFuns
@_delegate_statsfuns Normal norm μ σ

# `logerf(...)` is more accurate for arguments in the tails than `logsubexp(logcdf(...), logcdf(...))`
function logdiffcdf(d::Normal, x::Real, y::Real)
    x < y && throw(ArgumentError("requires x >= y."))
    μ, σ = params(d)
    _x, _y, _μ, _σ = promote(x, y, μ, σ)
    s = sqrt2 * _σ
    return logerf((_y - _μ) / s, (_x - _μ) / s) - logtwo
end

gradlogpdf(d::Normal, x::Real) = (d.μ - x) / d.σ^2

mgf(d::Normal, t::Real) = exp(t * d.μ + d.σ^2 / 2 * t^2)
function cgf(d::Normal, t)
    μ,σ = params(d)
    t*μ + (σ*t)^2/2
end
cf(d::Normal, t::Real) = exp(im * t * d.μ - d.σ^2 / 2 * t^2)

#### Affine transformations

Base.:+(d::Normal, c::Real) = Normal(d.μ + c, d.σ)
Base.:*(c::Real, d::Normal) = Normal(c * d.μ, abs(c) * d.σ)

#### Sampling

xval(d::Normal, z::Real) = muladd(d.σ, z, d.μ)

rand(rng::AbstractRNG, d::Normal{T}) where {T} = xval(d, randn(rng, float(T)))
function rand!(rng::AbstractRNG, d::Normal, A::AbstractArray{<:Real})
    randn!(rng, A)
    map!(Base.Fix1(xval, d), A, A)
    return A
end

#### Fitting

struct NormalStats{T<:Real} <: SufficientStats
    s::T    # (weighted) sum of x
    m::T    # (weighted) mean of x
    s2::T   # (weighted) sum of (x - μ)^2
    tw::T    # total sample weight
    function NormalStats(s::T1, m::T2, s2::T3, tw::T4) where {T1,T2,T3,T4}
        T = promote_type(T1, T2, T3, T4)
        return new{T}(T(s), T(m), T(s2), T(tw))
    end
end

function suffstats(::Type{<:Normal}, x::AbstractArray{T}) where T<:Real
    n = length(x)

    # compute s
    s = zero(T)
    for i in eachindex(x)
        s += x[i]
    end
    m = s / n

    # compute s2
    s2 = zero(T)
    for i in eachindex(x)
        s2 += abs2(x[i] - m)
    end

    NormalStats(s, m, s2, n)
end

function suffstats(::Type{<:Normal}, x::AbstractArray{T1}, w::AbstractArray{T2}) where {T1<:Real,T2<:Real}
    T = promote_type(T1, T2)
    # compute s
    tw = zero(T)
    s = zero(T)
    for i in eachindex(x, w)
        wi = w[i]
        s += wi * x[i]
        tw += wi
    end
    m = s / tw

    # compute s2
    s2 = zero(T)
    for i in eachindex(x, w)
        s2 += w[i] * abs2(x[i] - m)
    end

    NormalStats(s, m, s2, tw)
end

# Cases where μ or σ is known

struct NormalKnownMu{T<:Real} <: IncompleteDistribution
    μ::T
end

struct NormalKnownMuStats{T<:Real} <: SufficientStats
    μ::T      # known mean
    s2::T     # (weighted) sum of (x - μ)^2
    tw::T     # total sample weight
    function NormalKnownMuStats(μ::T1, s2::T2, tw::T3) where {T1,T2,T3}
        T = promote_type(T1, T2, T3)
        return new{T}(μ, s2, tw)
    end
end

function suffstats(g::NormalKnownMu{T0}, x::AbstractArray{T1}) where {T0,T1<:Real}
    T = promote_type(T0, T1)
    μ = g.μ
    s2 = zero(T)
    for i in eachindex(x)
        s2 += abs2(x[i] - μ)
    end
    NormalKnownMuStats(g.μ, s2, length(x))
end

function suffstats(g::NormalKnownMu{T0}, x::AbstractArray{T1}, w::AbstractArray{T2}) where {T0,T1<:Real,T2<:Real}
    T = promote_type(T0, T1, T2)
    μ = g.μ
    s2 = zero(T)
    tw = zero(T)
    for i in eachindex(x, w)
        wi = w[i]
        s2 += abs2(x[i] - μ) * wi
        tw += wi
    end
    NormalKnownMuStats(g.μ, s2, tw)
end

struct NormalKnownSigma{T<:Real} <: IncompleteDistribution
    σ::T
    function NormalKnownSigma(σ::T) where {T}
        σ > 0 || throw(ArgumentError("σ must be a positive value."))
        return new{T}(σ)
    end
end

struct NormalKnownSigmaStats{T<:Real} <: SufficientStats
    σ::T      # known std.dev
    sx::T      # (weighted) sum of x
    tw::T     # total sample weight
    function NormalKnownSigmaStats(σ::T1, sx::T2, tw::T3) where {T1,T2,T3}
        T = promote_type(T1, T2, T3)
        return new{T}(σ, sx, tw)
    end
end

function suffstats(g::NormalKnownSigma, x::AbstractArray{<:Real})
    NormalKnownSigmaStats(g.σ, sum(x), length(x))
end

function suffstats(g::NormalKnownSigma, x::AbstractArray{<:Real}, w::AbstractArray{<:Real})
    NormalKnownSigmaStats(g.σ, dot(x, w), sum(w))
end

# fit_mle based on sufficient statistics

fit_mle(::Type{D}, ss::NormalStats) where {D<:Normal} = D(ss.m, sqrt(ss.s2 / ss.tw))
fit_mle(g::NormalKnownMu, ss::NormalKnownMuStats) = Normal(g.μ, sqrt(ss.s2 / ss.tw))
fit_mle(g::NormalKnownSigma, ss::NormalKnownSigmaStats) = Normal(ss.sx / ss.tw, g.σ)

# generic fit_mle methods

function fit_mle(
    ::Type{D}, x::AbstractArray{<:Real};
    mu::Union{Nothing,<:Real}=nothing, sigma::Union{Nothing,<:Real}=nothing
) where {D<:Normal}
    if isnothing(mu)
        if isnothing(sigma)
            fit_mle(D, suffstats(Normal, x))
        else
            g = NormalKnownSigma(sigma)
            convert(D, fit_mle(g, suffstats(g, x)))
        end
    else
        if isnothing(sigma)
            g = NormalKnownMu(mu)
            convert(D, fit_mle(g, suffstats(g, x)))
        else
            D(mu, sigma)
        end
    end
end

function fit_mle(
    ::Type{D}, x::AbstractArray{<:Real}, w::AbstractArray{<:Real};
    mu::Union{Nothing,<:Real}=nothing, sigma::Union{Nothing,<:Real}=nothing
) where {D<:Normal}
    if isnothing(mu)
        if isnothing(sigma)
            fit_mle(D, suffstats(Normal, x, w))
        else
            g = NormalKnownSigma(sigma)
            convert(D, fit_mle(g, suffstats(g, x, w)))
        end
    else
        if isnothing(sigma)
            g = NormalKnownMu(mu)
            convert(D, fit_mle(g, suffstats(g, x, w)))
        else
            D(mu, sigma)
        end
    end
end
