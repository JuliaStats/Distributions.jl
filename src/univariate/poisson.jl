immutable Poisson <: DiscreteUnivariateDistribution
    lambda::Float64
    function Poisson(l::Real)
    	l > zero(l) || error("lambda must be positive")
        new(float64(l))
    end
    Poisson() = new(1.0)
end

@_jl_dist_1p Poisson pois


isupperbounded(::Union(Poisson, Type{Poisson})) = false
islowerbounded(::Union(Poisson, Type{Poisson})) = true
isbounded(::Union(Poisson, Type{Poisson})) = false

minimum(::Union(Poisson, Type{Poisson})) = 0
maximum(::Union(Poisson, Type{Poisson})) = Inf

insupport(::Poisson, x::Real) = isinteger(x) && zero(x) <= x
insupport(::Type{Poisson}, x::Real) = isinteger(x) && zero(x) <= x


function probs(d::Poisson, rgn::UnitRange)
    λ = d.lambda
    f, l = rgn[1], rgn[end]
    0 <= f <= l || throw(BoundsError())
    r = Array(Float64, l - f + 1)
    v = r[1] = pdf(d, f)
    if l > f
        b = f - 1
        for x = f+1:l
            c = λ / x
            r[x - b] = (v *= c)
        end
    end
    return r
end

mean(d::Poisson) = d.lambda

median(d::Poisson) = quantile(d, 0.5)

mode(d::Poisson) = ifloor(d.lambda)
modes(d::Poisson) = isinteger(d.lambda) ? [int(d.lambda)-1,int(d.lambda)] : [ifloor(d.lambda)]

var(d::Poisson) = d.lambda

skewness(d::Poisson) = 1.0 / sqrt(d.lambda)

kurtosis(d::Poisson) = 1.0 / d.lambda

function entropy(d::Poisson)
    λ = d.lambda
    if λ < 50.0
        s = 0.0
        for k in 1:100
            s += λ^k * lgamma(k + 1.0) / gamma(k + 1.0)
        end
        return λ * (1.0 - log(λ)) + exp(-λ) * s
    else
        return 0.5 * log(2 * pi * e * λ) -
               (1 / (12 * λ)) -
               (1 / (24 * λ * λ)) -
               (19 / (360 * λ * λ * λ))
    end
end

function mgf(d::Poisson, t::Real)
    l = d.lambda
    return exp(l * (exp(t) - 1.0))
end

function cf(d::Poisson, t::Real)
    l = d.lambda
    return exp(l * (exp(im * t) - 1.0))
end

# model fitting

immutable PoissonStats <: SufficientStats
    sx::Float64   # (weighted) sum of x
    tw::Float64   # total sample weight
end

suffstats(::Type{Poisson}, x::Array) = PoissonStats(float64(sum(x)), float64(length(x)))

function suffstats(::Type{Poisson}, x::Array, w::Array{Float64})
    n = length(x)
    n == length(w) || throw(ArgumentError("Inconsistent array lengths."))
    sx = 0.
    tw = 0.
    for i = 1 : n
        @inbounds wi = w[i]
        @inbounds sx += x[i] * wi
        tw += wi
    end
    PoissonStats(sx, tw)
end

fit_mle(::Type{Poisson}, ss::PoissonStats) = Poisson(ss.sx / ss.tw)

