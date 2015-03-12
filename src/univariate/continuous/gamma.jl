immutable Gamma <: ContinuousUnivariateDistribution
    α::Float64
    β::Float64

    function Gamma(α::Real, β::Real)
        α > zero(α) && β > zero(β) || error("Gamma: both shape and scale must be positive")
        @compat new(Float64(α), Float64(β))
    end

    Gamma(α::Real) = Gamma(α, 1.0)
    Gamma() = new(1.0, 1.0)
end

@_jl_dist_2p Gamma gamma

@distr_support Gamma 0.0 Inf


#### Parameters

shape(d::Gamma) = d.α
scale(d::Gamma) = d.β
rate(d::Gamma) = 1.0 / d.β

params(d::Gamma) = (d.α, d.β)


#### Statistics

mean(d::Gamma) = d.α * d.β

var(d::Gamma) = d.α * d.β^2

skewness(d::Gamma) = 2.0 / sqrt(d.α)

kurtosis(d::Gamma) = 6.0 / d.α

function mode(d::Gamma)
    (α, β) = params(d)
    α >= 1.0 ? β * (α - 1.0) : error("Gamma has no mode when shape < 1.0")
end

function entropy(d::Gamma)
    (α, β) = params(d)
    α + lgamma(α) + (1.0 - α) * digamma(α) + log(β)
end

mgf(d::Gamma, t::Real) = (1.0 - t * d.β)^(-d.α)

cf(d::Gamma, t::Real) = (1.0 - im * t * d.β)^(-d.α)


#### Evaluation

gradlogpdf(d::Gamma, x::Float64) =
    insupport(Gamma, x) ? (d.α - 1.0) / x - 1.0 / d.β : 0.0


#### Fit model

immutable GammaStats <: SufficientStats
    sx::Float64      # (weighted) sum of x
    slogx::Float64   # (weighted) sum of log(x)
    tw::Float64      # total sample weight

    @compat GammaStats(sx::Real, slogx::Real, tw::Real) = new(Float64(sx), Float64(slogx), Float64(tw))
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
