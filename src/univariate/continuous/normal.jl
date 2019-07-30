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
    Normal{T}(µ::T, σ::T) where {T<:Real} = new{T}(µ, σ)
end

function Normal(μ::T, σ::T) where {T <: Real}
    @check_args(Normal, σ >= zero(σ))
    return Normal{T}(μ, σ)
end

#### Outer constructors
Normal(μ::T, σ::T, ::NoArgCheck) where {T<:Real} = Normal{T}(μ, σ)
Normal(μ::Real, σ::Real) = Normal(promote(μ, σ)...)
Normal(μ::Integer, σ::Integer) = Normal(float(μ), float(σ))
Normal(μ::T) where {T <: Real} = Normal(μ, one(T))
Normal() = Normal(0.0, 1.0, NoArgCheck())

const Gaussian = Normal

# #### Conversions
convert(::Type{Normal{T}}, μ::S, σ::S) where {T <: Real, S <: Real} = Normal(T(μ), T(σ))
convert(::Type{Normal{T}}, d::Normal{S}) where {T <: Real, S <: Real} = Normal(T(d.μ), T(d.σ), NoArgCheck())

@distr_support Normal -Inf Inf

#### Parameters

params(d::Normal) = (d.μ, d.σ)
@inline partype(d::Normal{T}) where {T<:Real} = T

location(d::Normal) = d.μ
scale(d::Normal) = d.σ

eltype(::Normal{T}) where {T} = T

#### Statistics

mean(d::Normal) = d.μ
median(d::Normal) = d.μ
mode(d::Normal) = d.μ

var(d::Normal) = abs2(d.σ)
std(d::Normal) = d.σ
skewness(d::Normal{T}) where {T<:Real} = zero(T)
kurtosis(d::Normal{T}) where {T<:Real} = zero(T)

entropy(d::Normal) = (log2π + 1)/2 + log(d.σ)

#### Evaluation

# Helpers
"""
    xval(d::Normal, z::Real)

Computes the x-value based on a Normal distribution and a z-value.
"""
function xval(d::Normal, z::Real)
    if isinf(z) && iszero(d.σ)
        d.μ + one(d.σ) * z
    else
        d.μ + d.σ * z
    end
end
"""
    zval(d::Normal, x::Real)

Computes the z-value based on a Normal distribution and a x-value.
"""
zval(d::Normal, x::Real) = (x - d.μ) / d.σ

gradlogpdf(d::Normal, x::Real) = -zval(d, x) / d.σ
# logpdf
function logpdf(d::Normal, x::Real)
    if iszero(d.σ)
        d.μ == x ? Inf : -Inf
    else
        -(zval(d, x)^2 + log2π) / 2 - log(d.σ)
    end
end
# pdf
function pdf(d::Normal, x::Real)
    if iszero(d.σ)
        d.μ == x ? Inf : 0.0
    else
        exp(-zval(d, x)^2 / 2) * invsqrt2π / d.σ
    end
end
# logcdf
function logcdf(d::Normal, x::Real)
    if iszero(d.σ)
        d.μ ≤ x ? 0.0 : -Inf
    else
        z = zval(d, x)
        if z < -1.0
            log(erfcx(-z * invsqrt2)/2) - abs2(z)/2
        else
            log1p(-erfc(z * invsqrt2)/2)
        end
    end
end
# logccdf
function logccdf(d::Normal, x::Real)
    if iszero(d.σ) && x == d.μ
        z = zval(Normal(zero(d.μ), d.σ), one(x))
    else
        z = zval(d, x)
    end
    z > 1.0 ?
    log(erfcx(z * invsqrt2) / 2) - z^2 / 2 :
    log1p(-erfc(-z * invsqrt2) / 2)
end
# cdf
function cdf(d::Normal, x::Real)
    if iszero(d.σ)
        float(d.μ ≤ x)
    else
        erfc(-zval(d, x) * invsqrt2) / 2
    end
end
# ccdf
function ccdf(d::Normal, x::Real)
    z = iszero(d.σ) && x == d.μ ?
        zval(Normal(zero(d.μ), d.σ), one(x)) :
        zval(d, x)
    erfc(z * invsqrt2) / 2
end
# invlogcdf
"""
    norminvlogcdf(lp::Real)

Helper function that calls `_norminvlogcdf_impl` used for `invlogccdf` with the Normal distributions.
"""
norminvlogcdf(lp::Real) = _norminvlogcdf_impl(convert(Float64, lp))
norminvlogcdf(lp::Union{Float16,Float32}) = convert(typeof(lp), _norminvlogcdf_impl(convert(Float64, lp)))
invlogcdf(d::Normal, lp::Real) = xval(d, norminvlogcdf(lp))
# invlogccdf
invlogccdf(d::Normal, lp::Real) = xval(d, -norminvlogcdf(lp))
# quantile
function quantile(d::Normal, p::Real)
    if iszero(d.σ)
        if iszero(p)
            -Inf
        elseif isone(p)
            Inf
        else
            0.5
        end
    end
    xval(d, -erfcinv(2p) * sqrt2)
end
# cquantile
function cquantile(d::Normal, q::Real)
    if iszero(d.σ)
        if iszero(q)
            Inf
        elseif isone(q)
            -Inf
        else
            0.5
        end
    end
    xval(d, erfcinv(2q) * sqrt2)
end

# norminvcdf & norminvlogcdf implementation
"""
    _norminvlogcdf_impl(lp::Float64)

Uses `_qnorm_ker1` and `_qnorm_ker2` to obtain the ational approximations for the inverse cdf and its logarithm, from:
Wichura, M.J. (1988) Algorithm AS 241: The Percentage Points of the Normal Distribution
  Journal of the Royal Statistical Society. Series C (Applied Statistics), Vol. 37, No. 3, pp. 477-484
"""
function _norminvlogcdf_impl(lp::Float64)
    if isfinite(lp) && lp < 0.0
        q = exp(lp) - 0.5
        # qnorm_kernel(lp, q, true)
        if abs(q) ≤ 0.425
            _qnorm_ker1(q)
        else
            r = sqrt(q < 0 ? -lp : -log1mexp(lp))
            return copysign(_qnorm_ker2(r), q)
        end
    elseif lp ≥ 0.0
        iszero(lp) ? Inf : NaN
    else # lp is -Inf or NaN
        lp
    end
end
"""
    _qnorm_ker1(r::Float64)

Along with `_qnorm_ker2` these helpers enable the rational approximations for the inverse cdf and its logarithm
"""
function _qnorm_ker1(q::Float64)
    # pre-condition: abs(q) ≤ 0.425
    r = 0.180625 - q^2
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
"""
    _qnorm_ker2(r::Float64)

Along with `_qnorm_ker1` these helpers enable the rational approximations for the inverse cdf and its logarithm
"""
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

mgf(d::Normal, t::Real) = exp(t * d.μ + d.σ^2 / 2 * t^2)
cf(d::Normal, t::Real) = exp(im * t * d.μ - d.σ^2 / 2 * t^2)

#### Sampling

rand(rng::AbstractRNG, d::Normal{T}) where {T} = d.μ + d.σ * randn(rng, T)

#### Fitting

struct NormalStats <: SufficientStats
    s::Float64    # (weighted) sum of x
    m::Float64    # (weighted) mean of x
    s2::Float64   # (weighted) sum of (x - μ)^2
    tw::Float64    # total sample weight
end

function suffstats(::Type{<:Normal}, x::AbstractArray{T}) where T<:Real
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

function suffstats(::Type{<:Normal}, x::AbstractArray{T}, w::AbstractArray{Float64}) where T<:Real
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

fit_mle(::Type{<:Normal}, ss::NormalStats) = Normal(ss.m, sqrt(ss.s2 / ss.tw))
fit_mle(g::NormalKnownMu, ss::NormalKnownMuStats) = Normal(g.μ, sqrt(ss.s2 / ss.tw))
fit_mle(g::NormalKnownSigma, ss::NormalKnownSigmaStats) = Normal(ss.sx / ss.tw, g.σ)

# generic fit_mle methods

function fit_mle(::Type{<:Normal}, x::AbstractArray{T}; mu::Float64=NaN, sigma::Float64=NaN) where T<:Real
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

function fit_mle(::Type{<:Normal}, x::AbstractArray{T}, w::AbstractArray{Float64}; mu::Float64=NaN, sigma::Float64=NaN) where T<:Real
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
