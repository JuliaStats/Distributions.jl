immutable Gamma <: ContinuousUnivariateDistribution
    shape::Float64
    scale::Float64

    function Gamma(sh::Real, sc::Real)
        sh > zero(sh) && sc > zero(sc) || 
            error("Both shape and scale must be positive")
        new(float64(sh), float64(sc))
    end

    Gamma(sh::Real) = Gamma(sh, 1.0)
    Gamma() = Gamma(1.0, 1.0)
end

scale(d::Gamma) = d.scale
rate(d::Gamma) = 1.0 / d.scale

@_jl_dist_2p Gamma gamma

@continuous_distr_support Gamma 0.0 Inf

function entropy(d::Gamma)
    x = (1.0 - d.shape) * digamma(d.shape)
    x + lgamma(d.shape) + log(d.scale) + d.shape
end

kurtosis(d::Gamma) = 6.0 / d.shape

mean(d::Gamma) = d.shape * d.scale

median(d::Gamma) = quantile(d, 0.5)

mgf(d::Gamma, t::Real) = (1.0 - t * d.scale)^(-d.shape)

cf(d::Gamma, t::Real) = (1.0 - im * t * d.scale)^(-d.shape)

function mode(d::Gamma)
    d.shape >= 1.0 ? d.scale * (d.shape - 1.0) : error("Gamma has no mode when shape < 1.0")
end

modes(d::Gamma) = [mode(d)]

rand(d::Gamma) = d.scale * randg(d.shape)

function rand!(d::Gamma, A::Array{Float64})
    s = GammaSampler(d.shape)
    for i = 1:length(A)
        A[i] = rand(s)
    end
    multiply!(A, d.scale)
end

skewness(d::Gamma) = 2.0 / sqrt(d.shape)

var(d::Gamma) = d.shape * d.scale * d.scale

## Fit model

immutable GammaStats <: SufficientStats
    sx::Float64      # (weighted) sum of x
    slogx::Float64   # (weighted) sum of log(x)
    tw::Float64      # total sample weight

    GammaStats(sx::Real, slogx::Real, tw::Real) = new(float64(sx), float64(slogx), float64(tw))
end

function suffstats(::Type{Gamma}, x::Array)
    sx = 0.
    slogx = 0.
    for xi = x
        sx += xi
        slogx += log(xi)
    end
    GammaStats(sx, slogx, length(x))
end

function suffstats(::Type{Gamma}, x::Array, w::Array{Float64})
    n = length(x)
    if length(w) != n
        throw(ArgumentError("Inconsistent argument dimensions."))
    end

    sx = 0.
    slogx = 0.
    tw = 0.
    for i = 1:n
        @inbounds xi = x[i]
        @inbounds wi = w[i]
        sx += wi * xi
        slogx += wi * log(xi)
        tw += wi
    end
    GammaStats(sx, slogx, tw)
end

function gamma_mle_update(logmx::Float64, mlogx::Float64, a::Float64)
    ia = 1.0 / a
    z = ia + (mlogx - logmx + log(a) - digamma(a)) / (abs2(a) * (ia - trigamma(a)))
    1.0 / z
end

function fit_mle(::Type{Gamma}, ss::GammaStats; 
    alpha0::Float64=NaN, maxiter::Int=1000, tol::Float64=1.0e-16)

    mx = ss.sx / ss.tw
    logmx = log(mx)
    mlogx = ss.slogx / ss.tw

    a::Float64 = isnan(alpha0) ? 0.5 / (logmx - mlogx) : alpha0
    converged = false
    
    t = 0
    while !converged && t < maxiter
        t += 1
        a_old = a
        a = gamma_mle_update(logmx, mlogx, a)
        converged = abs(a - a_old) <= tol
    end

    Gamma(a, mx / a)
end
