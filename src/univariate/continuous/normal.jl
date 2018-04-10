"""
    Normal(μ,σ)

The *Normal distribution* with mean `μ` and standard deviation `σ` has probability density function

```math
f(x; \\mu, \\sigma) = \\frac{1}{\\sqrt{2 \\pi \\sigma^2}}
\\exp \\left( - \\frac{(x - \\mu)^2}{2 \\sigma^2} \\right)
```

```julia
Normal()          # standard Normal distribution with zero mean and unit variance
Normal(mu)        # Normal distribution with mean mu and unit variance
Normal(mu, sig)   # Normal distribution with mean mu and variance sig^2

params(d)         # Get the parameters, i.e. (mu, sig)
mean(d)           # Get the mean, i.e. mu
std(d)            # Get the standard deviation, i.e. sig
```

External links

* [Normal distribution on Wikipedia](http://en.wikipedia.org/wiki/Normal_distribution)

"""
struct Normal{T<:Real} <: ContinuousUnivariateDistribution
    μ::T
    σ::T

    Normal{T}(μ, σ) where {T} = (@check_args(Normal, σ > zero(σ)); new{T}(μ, σ))
end

#### Outer constructors
Normal(μ::T, σ::T) where {T<:Real} = Normal{T}(μ, σ)
Normal(μ::Real, σ::Real) = Normal(promote(μ, σ)...)
Normal(μ::Integer, σ::Integer) = Normal(Float64(μ), Float64(σ))
Normal(μ::Real) = Normal(μ, 1.0)
Normal() = Normal(0.0, 1.0)

const Gaussian = Normal

# #### Conversions
convert(::Type{Normal{T}}, μ::S, σ::S) where {T <: Real, S <: Real} = Normal(T(μ), T(σ))
convert(::Type{Normal{T}}, d::Normal{S}) where {T <: Real, S <: Real} = Normal(T(d.μ), T(d.σ))

@distr_support Normal -Inf Inf


#### Parameters

params(d::Normal) = (d.μ, d.σ)
@inline partype(d::Normal{T}) where {T<:Real} = T

location(d::Normal) = d.μ
scale(d::Normal) = d.σ

#### Statistics

mean(d::Normal) = d.μ
median(d::Normal) = d.μ
mode(d::Normal) = d.μ

var(d::Normal) = abs2(d.σ)
std(d::Normal) = d.σ
skewness(d::Normal{T}) where {T<:Real} = zero(T)
kurtosis(d::Normal{T}) where {T<:Real} = zero(T)

entropy(d::Normal) = (log2π + 1)/2 + log(d.σ)


#### Evaluation (see JuliaStats/Distributions.jl/issues/708)

xval(μ::Real, σ::Real, z::Number) = μ + σ * z
zval(μ::Real, σ::Real, x::Number) = (x - μ) / σ

# pdf
normpdf(z::Number) = exp(-abs2(z)/2) * invsqrt2π
normpdf(μ::Real, σ::Real, x::Number) = normpdf(zval(μ, σ, x)) / σ

# logpdf
normlogpdf(z::Number) = -(abs2(z) + log2π)/2
normlogpdf(μ::Real, σ::Real, x::Number) = normlogpdf(zval(μ, σ, x)) - log(σ)

# cdf
normcdf(z::Number) = erfc(-z * invsqrt2)/2
normcdf(μ::Real, σ::Real, x::Number) = normcdf(zval(μ, σ, x))

# ccdf
normccdf(z::Number) = erfc(z * invsqrt2)/2
normccdf(μ::Real, σ::Real, x::Number) = normccdf(zval(μ, σ, x))

# logcdf
normlogcdf(z::Number) = z < -1.0 ?
    log(erfcx(-z * invsqrt2)/2) - abs2(z)/2 :
    log1p(-erfc(z * invsqrt2)/2)
normlogcdf(μ::Real, σ::Real, x::Number) = normlogcdf(zval(μ, σ, x))

# logccdf
normlogccdf(z::Number) = z > 1.0 ?
    log(erfcx(z * invsqrt2)/2) - abs2(z)/2 :
    log1p(-erfc(-z * invsqrt2)/2)
normlogccdf(μ::Real, σ::Real, x::Number) = normlogccdf(zval(μ, σ, x))

norminvcdf(p::Real) = -erfcinv(2*p) * sqrt2
norminvcdf(μ::Real, σ::Real, p::Real) = xval(μ, σ, norminvcdf(p))

norminvccdf(p::Real) = erfcinv(2*p) * sqrt2
norminvccdf(μ::Real, σ::Real, p::Real) = xval(μ, σ, norminvccdf(p))

# invlogcdf. Fixme! Support more precisions than Float64
norminvlogcdf(lp::Union{Float16,Float32}) = convert(typeof(lp), _norminvlogcdf_impl(Float64(lp)))
norminvlogcdf(lp::Real) = _norminvlogcdf_impl(Float64(lp))
norminvlogcdf(μ::Real, σ::Real, lp::Real) = xval(μ, σ, norminvlogcdf(lp))

# invlogccdf. Fixme! Support more precisions than Float64
norminvlogccdf(lp::Union{Float16,Float32}) = convert(typeof(lp), -_norminvlogcdf_impl(Float64(lp)))
norminvlogccdf(lp::Real) = -_norminvlogcdf_impl(Float64(lp))
norminvlogccdf(μ::Real, σ::Real, lp::Real) = xval(μ, σ, norminvlogccdf(lp))


# norminvcdf & norminvlogcdf implementation
#
#   Rational approximations for the inverse cdf and its logarithm, from:
#
#   Wichura, M.J. (1988) Algorithm AS 241: The Percentage Points of the Normal Distribution
#   Journal of the Royal Statistical Society. Series C (Applied Statistics), Vol. 37, No. 3, pp. 477-484
#

function _norminvlogcdf_impl(lp::Float64)
    if isfinite(lp) && lp < 0.0
        q = exp(lp) - 0.5
        # qnorm_kernel(lp, q, true)
        if abs(q) <= 0.425
            _qnorm_ker1(q)
        else
            r = sqrt(q < 0 ? -lp : -log1mexp(lp))
            return copysign(_qnorm_ker2(r), q)
        end
    elseif lp >= 0.0
        lp == 0.0 ? Inf : NaN
    else # lp is -Inf or NaN
        lp
    end
end

function _qnorm_ker1(q::Float64)
    # pre-condition: abs(q) <= 0.425
    r = 0.180625 - q*q
    return q * @horner(r,
                       3.38713_28727_96366_6080e0,
                       1.33141_66789_17843_7745e2,
                       1.97159_09503_06551_4427e3,
                       1.37316_93765_50946_1125e4,
                       4.59219_53931_54987_1457e4,
                       6.72657_70927_00870_0853e4,
                       3.34305_75583_58812_8105e4,
                       2.50908_09287_30122_6727e3) /
    @horner(r,
            1.0,
            4.23133_30701_60091_1252e1,
            6.87187_00749_20579_0830e2,
            5.39419_60214_24751_1077e3,
            2.12137_94301_58659_5867e4,
            3.93078_95800_09271_0610e4,
            2.87290_85735_72194_2674e4,
            5.22649_52788_52854_5610e3)
end

function _qnorm_ker2(r::Float64)
    if r < 5.0
        r -= 1.6
        @horner(r,
                1.42343_71107_49683_57734e0,
                4.63033_78461_56545_29590e0,
                5.76949_72214_60691_40550e0,
                3.64784_83247_63204_60504e0,
                1.27045_82524_52368_38258e0,
                2.41780_72517_74506_11770e-1,
                2.27238_44989_26918_45833e-2,
                7.74545_01427_83414_07640e-4) /
        @horner(r,
                1.0,
                2.05319_16266_37758_82187e0,
                1.67638_48301_83803_84940e0,
                6.89767_33498_51000_04550e-1,
                1.48103_97642_74800_74590e-1,
                1.51986_66563_61645_71966e-2,
                5.47593_80849_95344_94600e-4,
                1.05075_00716_44416_84324e-9)
    else
        r -= 5.0
        @horner(r,
                6.65790_46435_01103_77720e0,
                5.46378_49111_64114_36990e0,
                1.78482_65399_17291_33580e0,
                2.96560_57182_85048_91230e-1,
                2.65321_89526_57612_30930e-2,
                1.24266_09473_88078_43860e-3,
                2.71155_55687_43487_57815e-5,
                2.01033_43992_92288_13265e-7) /
        @horner(r,
                1.0,
                5.99832_20655_58879_37690e-1,
                1.36929_88092_27358_05310e-1,
                1.48753_61290_85061_48525e-2,
                7.86869_13114_56132_59100e-4,
                1.84631_83175_10054_68180e-5,
                1.42151_17583_16445_88870e-7,
                2.04426_31033_89939_78564e-15)
    end
end

gradlogpdf(d::Normal, x::Real) = (d.μ - x) / d.σ^2

mgf(d::Normal, t::Real) = exp(t * d.μ + d.σ^2/2 * t^2)
cf(d::Normal, t::Real) = exp(im * t * d.μ - d.σ^2/2 * t^2)


#### Sampling

rand(d::Normal) = rand(GLOBAL_RNG, d)
rand(rng::AbstractRNG, d::Normal) = d.μ + d.σ * randn(rng)


#### Fitting

struct NormalStats <: SufficientStats
    s::Float64    # (weighted) sum of x
    m::Float64    # (weighted) mean of x
    s2::Float64   # (weighted) sum of (x - μ)^2
    tw::Float64    # total sample weight
end

function suffstats(::Type{Normal}, x::AbstractArray{T}) where T<:Real
    n = length(x)

    # compute s
    s = x[1]
    for i = 2:n
        @inbounds s += x[i]
    end
    m = s / n

    # compute s2
    s2 = abs2(x[1] - m)
    for i = 2:n
        @inbounds s2 += abs2(x[i] - m)
    end

    NormalStats(s, m, s2, n)
end

function suffstats(::Type{Normal}, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Real
    n = length(x)

    # compute s
    tw = w[1]
    s = w[1] * x[1]
    for i = 2:n
        @inbounds wi = w[i]
        @inbounds s += wi * x[i]
        tw += wi
    end
    m = s / tw

    # compute s2
    s2 = w[1] * abs2(x[1] - m)
    for i = 2:n
        @inbounds s2 += w[i] * abs2(x[i] - m)
    end

    NormalStats(s, m, s2, tw)
end

# Cases where μ or σ is known

struct NormalKnownMu <: IncompleteDistribution
    μ::Float64
end

struct NormalKnownMuStats <: SufficientStats
    μ::Float64      # known mean
    s2::Float64     # (weighted) sum of (x - μ)^2
    tw::Float64     # total sample weight
end

function suffstats(g::NormalKnownMu, x::AbstractArray{T}) where T<:Real
    μ = g.μ
    s2 = abs2(x[1] - μ)
    for i = 2:length(x)
        @inbounds s2 += abs2(x[i] - μ)
    end
    NormalKnownMuStats(g.μ, s2, length(x))
end

function suffstats(g::NormalKnownMu, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Real
    μ = g.μ
    s2 = abs2(x[1] - μ) * w[1]
    tw = w[1]
    for i = 2:length(x)
        @inbounds wi = w[i]
        @inbounds s2 += abs2(x[i] - μ) * wi
        tw += wi
    end
    NormalKnownMuStats(g.μ, s2, tw)
end


struct NormalKnownSigma <: IncompleteDistribution
    σ::Float64

    function NormalKnownSigma(σ::Float64)
        σ > 0 || throw(ArgumentError("σ must be a positive value."))
        new(σ)
    end
end

struct NormalKnownSigmaStats <: SufficientStats
    σ::Float64      # known std.dev
    sx::Float64      # (weighted) sum of x
    tw::Float64     # total sample weight
end

function suffstats(g::NormalKnownSigma, x::AbstractArray{T}) where T<:Real
    NormalKnownSigmaStats(g.σ, sum(x), Float64(length(x)))
end

function suffstats(g::NormalKnownSigma, x::AbstractArray{T}, w::AbstractArray{T}) where T<:Real
    NormalKnownSigmaStats(g.σ, dot(x, w), sum(w))
end

# fit_mle based on sufficient statistics

fit_mle(::Type{Normal}, ss::NormalStats) = Normal(ss.m, sqrt(ss.s2 / ss.tw))
fit_mle(g::NormalKnownMu, ss::NormalKnownMuStats) = Normal(g.μ, sqrt(ss.s2 / ss.tw))
fit_mle(g::NormalKnownSigma, ss::NormalKnownSigmaStats) = Normal(ss.sx / ss.tw, g.σ)

# generic fit_mle methods

function fit_mle(::Type{Normal}, x::AbstractArray{T}; mu::Float64=NaN, sigma::Float64=NaN) where T<:Real
    if isnan(mu)
        if isnan(sigma)
            fit_mle(Normal, suffstats(Normal, x))
        else
            g = NormalKnownSigma(sigma)
            fit_mle(g, suffstats(g, x))
        end
    else
        if isnan(sigma)
            g = NormalKnownMu(mu)
            fit_mle(g, suffstats(g, x))
        else
            Normal(mu, sigma)
        end
    end
end

function fit_mle(::Type{Normal}, x::AbstractArray{T}, w::AbstractArray{Float64}; mu::Float64=NaN, sigma::Float64=NaN) where T<:Real
    if isnan(mu)
        if isnan(sigma)
            fit_mle(Normal, suffstats(Normal, x, w))
        else
            g = NormalKnownSigma(sigma)
            fit_mle(g, suffstats(g, x, w))
        end
    else
        if isnan(sigma)
            g = NormalKnownMu(mu)
            fit_mle(g, suffstats(g, x, w))
        else
            Normal(mu, sigma)
        end
    end
end
