#import Base.Math.@horner
macro horner(x, p...)
    ex = p[end]
    for i = length(p)-1:-1:1
        ex = :($(p[i]) + $x * $ex)
    end
    ex
end


immutable Normal <: ContinuousUnivariateDistribution
    μ::Float64
    σ::Float64
    function Normal(μ::Real, σ::Real)
    	σ > zero(σ) || error("std.dev. must be positive")
    	new(float64(μ), float64(σ))
    end
end
Normal(μ::Real) = Normal(float64(μ), 1.0)
Normal() = Normal(0.0, 1.0)

@_jl_dist_2p Normal norm

const Gaussian = Normal

zval(d::Normal, x::Real) = (x - d.μ)/d.σ
xval(d::Normal, z::Real) = d.μ + d.σ * z

φ(z::Real) = exp(-0.5*z*z)/√2π
pdf(d::Normal, x::Real) = φ(zval(d,x))/d.σ

logφ(z::Real) = -0.5*(z*z + log2π)
logpdf(d::Normal, x::Real) = logφ(zval(d,x)) - log(d.σ)

Φ(z::Real) = 0.5*erfc(-z/√2)
cdf(d::Normal, x::Real) = Φ(zval(d,x))

Φc(z::Real) = 0.5*erfc(z/√2)
ccdf(d::Normal, x::Real) = Φc(zval(d,x))

logΦ(z::Real) = z < -1.0 ? log(0.5*erfcx(-z/√2)) - 0.5*z*z : log1p(-0.5*erfc(z/√2))
logcdf(d::Normal, x::Real) = logΦ(zval(d,x))

logΦc(z::Real) = z > 1.0 ? log(0.5*erfcx(z/√2)) - 0.5*z*z : log1p(-0.5*erfc(-z/√2))
logccdf(d::Normal, x::Real) = logΦc(zval(d,x))    

# Rational approximations for the inverse cdf, from:
#   Wichura, M.J. (1988) Algorithm AS 241: The Percentage Points of the Normal Distribution
#   Journal of the Royal Statistical Society. Series C (Applied Statistics), Vol. 37, No. 3, pp. 477-484
for (fn,arg) in ((:Φinv,:p),(:logΦinv,:logp))
    @eval begin
        function $fn($arg::Real)
            if $(fn == :Φinv)
                q = p - 0.5
            else
                q = exp(logp) - 0.5
            end
            if abs(q) <= 0.425 
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
            else
                if $(fn == :Φinv)
                    if p <= 0.0
                        return p == 0.0 ? -inf(Float64) : nan(Float64)
                    elseif p >= 1.0 
                        return p == 1.0 ? inf(Float64) : nan(Float64)
                    end
                    r = sqrt(q < 0 ? -log(p) : -log1p(-p))
                else
                    if logp == -Inf
                        return -inf(Float64)
                    elseif logp >= 0.0 
                        return logp == 0.0 ? inf(Float64) : nan(Float64)
                    end
                    r = sqrt(q < 0 ? -logp : -log(-expm1(logp)))
                end
                if r < 5.0
                    r -= 1.6
                    z = @horner(r,
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
                    z = @horner(r,
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
                return copysign(z,q)
            end
        end
    end
end

quantile(d::Normal, p::Real) = xval(d, Φinv(p))
cquantile(d::Normal, p::Real) = xval(d, -Φinv(p))
invlogcdf(d::Normal, p::Real) = xval(d, logΦinv(p))
invlogccdf(d::Normal, p::Real) = xval(d, -logΦinv(p))


entropy(d::Normal) = 0.5 * (log2π + 1.) + log(d.σ)

insupport(d::Normal, x::Real) = isfinite(x)

kurtosis(d::Normal) = 0.0

mean(d::Normal) = d.μ

median(d::Normal) = d.μ

mgf(d::Normal, t::Real) = exp(t * d.μ + 0.5 * d.σ^t * t^2)

cf(d::Normal, t::Real) = exp(im * t * d.μ - 0.5 * d.σ^t * t^2)

mode(d::Normal) = d.μ
modes(d::Normal) = [d.μ]

rand(d::Normal) = d.μ + d.σ * randn()

skewness(d::Normal) = 0.0

std(d::Normal) = d.σ

var(d::Normal) = d.σ^2

## Fit model

immutable NormalStats
    s::Float64    # (weighted) sum of x
    m::Float64    # (weighted) mean of x
    s2::Float64   # (weighted) sum of (x - μ)^2
    tw::Float64    # total sample weight

    NormalStats(s::Real, m::Real, s2::Real, tw::Real) = new(float64(s), float64(m), float64(s2), float64(tw))
end

function suffstats{T<:Real}(::Type{Normal}, x::Array{T}) 
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

function suffstats{T<:Real}(::Type{Normal}, x::Array{T}, w::Array{Float64}) 
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
        @inbounds s2 += abs2(x[i] - m)
    end

    NormalStats(s, m, s2, tw)
end

# Cases where μ or σ is known

immutable NormalKnownMu <: GenerativeFormulation
    μ::Float64
end

immutable NormalKnownMuStats
    μ::Float64      # known mean
    s2::Float64     # (weighted) sum of (x - μ)^2
    tw::Float64     # total sample weight
end

function suffstats{T<:Real}(g::NormalKnownMu, x::Array{T})
    μ = g.μ
    s2 = abs2(x[1] - μ)
    for i = 2:length(x)
        @inbounds s2 += abs2(x[i] - μ)
    end
    NormalKnownMuStats(g.μ, s2, float64(length(x)))
end

function suffstats{T<:Real}(g::NormalKnownMu, x::Array{T}, w::Array{Float64})
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


immutable NormalKnownSigma <: GenerativeFormulation
    σ::Float64

    function NormalKnownSigma(σ::Float64)
        σ > 0.0 || throw(ArgumentError("σ must be a positive value."))
        new(σ)
    end
end

immutable NormalKnownSigmaStats
    σ::Float64      # known std.dev
    s::Float64      # (weighted) sum of x
    tw::Float64     # total sample weight
end

function suffstats{T<:Real}(g::NormalKnownSigma, x::Array{T})
    NormalKnownSigmaStats(g.σ, sum(x), float64(length(x)))    
end

function suffstats{T<:Real}(g::NormalKnownSigma, x::Array{T}, w::Array{T})
    NormalKnownSigmaStats(g.σ, dot(x, w), sum(w))    
end

# fit_mle based on sufficient statistics

fit_mle(::Type{Normal}, ss::NormalStats) = Normal(ss.m, sqrt(ss.s2 / ss.tw))
fit_mle(g::NormalKnownMu, ss::NormalKnownMuStats) = Normal(g.μ, ss.s2 / ss.tw)
fit_mle(g::NormalKnownSigma, ss::NormalKnownSigmaStats) = Normal(ss.s / ss.tw, g.σ)

# generic fit_mle methods

function fit_mle{T<:Real}(::Type{Normal}, x::Array{T}; mu::Float64=NaN, sigma::Float64=NaN)
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

function fit_mle{T<:Real}(::Type{Normal}, x::Array{T}, w::Array{Float64}; mu::Float64=NaN, sigma::Float64=NaN)
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

