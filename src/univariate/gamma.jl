immutable Gamma <: ContinuousUnivariateDistribution
    shape::Float64
    scale::Float64
    function Gamma(sh::Real, sc::Real)
        sh > zero(sh) && sc > zero(sc) || 
            error("Both shape and scale must be positive")
        new(float64(sh), float64(sc))
    end
end

Gamma(sh::Real) = Gamma(sh, 1.0)
Gamma() = Gamma(1.0, 1.0) # Standard exponential distribution

scale(d::Gamma) = d.scale
rate(d::Gamma) = 1.0 / d.scale

@_jl_dist_2p Gamma gamma

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

# rand()
#
#  A simple method for generating gamma variables - Marsaglia and Tsang (2000)
#  http://www.cparity.com/projects/AcmClassification/samples/358414.pdf
#  Page 369
#  basic simulation loop for pre-computed d and c
function randg2(d::Float64, c::Float64) 
    while true
        x = v = 0.0
        while v <= 0.0
            x = randn()
            v = 1.0 + c * x
        end
        v = v^3
        U = rand()
        x2 = x^2
        if U < 1.0 - 0.331 * x2^2 ||
           log(U) < 0.5 * x2 + d * (1.0 - v + log(v))
            return d * v
        end
    end
end


# sampling from Gamma(α, 1)
function randg(α::Float64)
    dpar = (α <= 1.0 ? α + 1.0 : α) - 1.0 / 3.0
    cpar = 1.0 / sqrt(9.0 * dpar)
    randg2(dpar, cpar) * (α > 1.0 ? 1.0 : rand()^(1.0 / α))
end

rand(d::Gamma) = d.scale * randg(d.shape)

function rand!(d::Gamma, A::Array{Float64})
    α = d.shape
    dpar = (α <= 1.0 ? α + 1.0 : α) - 1.0 / 3.0
    cpar = 1.0 / sqrt(9.0 * dpar)
    n = length(A)
    for i in 1:n
        A[i] = randg2(dpar, cpar)
    end
    if α <= 1.0
        ainv = 1.0 / α
        for i in 1:n
            A[i] *= rand()^ainv
        end
    end
    multiply!(A, d.scale)
end

skewness(d::Gamma) = 2.0 / sqrt(d.shape)

var(d::Gamma) = d.shape * d.scale * d.scale

### handling support
isupperbounded(::Union(Gamma, Type{Gamma})) = false
islowerbounded(::Union(Gamma, Type{Gamma})) = true
isbounded(::Union(Gamma, Type{Gamma})) = false

hasfinitesupport(::Union(Gamma, Type{Gamma})) = false
min(::Union(Gamma, Type{Gamma})) = zero(Real)
max(::Union(Gamma, Type{Gamma})) = Inf

insupport(::Union(Gamma, Type{Gamma}), x::Real) = min(Gamma) <= x < max(Gamma)

## Fit model

immutable GammaStats
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
