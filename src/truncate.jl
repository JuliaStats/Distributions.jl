
immutable Truncated{D<:UnivariateDistribution,S<:ValueSupport} <: Distribution{Univariate,S}
    untruncated::D
    lower::Float64
    upper::Float64
    nc::Float64
    function Truncated{T<:UnivariateDistribution}(d::T, l::Real, u::Real, nc::Real)
        if l >= u
            error("upper must be > lower")
        end
        new(d, float64(l), float64(u), float64(nc))
    end
end

function Truncated{S<:ValueSupport}(d::UnivariateDistribution{S}, l::Real, u::Real, nc::Real)
    Truncated{typeof(d),S}(d,l,u,nc)
end

function Truncated{S<:ValueSupport}(d::UnivariateDistribution{S}, l::Real, u::Real)
    Truncated{typeof(d),S}(d,l,u, cdf(d, u) - cdf(d, l))
end

function insupport(d::Truncated, x::Number)
    return x >= d.lower && x <= d.upper && insupport(d.untruncated, x)
end

function pdf(d::Truncated, x::Real)
    if !insupport(d, x)
        return 0.0
    else
        return pdf(d.untruncated, x) / d.nc
    end
end

function logpdf(d::Truncated, x::Real)
    if !insupport(d, x)
        return -Inf
    else
        return logpdf(d.untruncated, x) - log(d.nc)
    end
end

function cdf(d::Truncated, x::Real)
    if x < d.lower
        return 0.0
    elseif x > d.upper
        return 1.0
    else
        return (cdf(d.untruncated, x) - cdf(d.untruncated, d.lower)) / d.nc
    end
end

function quantile(d::Truncated, p::Real)
    top = cdf(d.untruncated, d.upper)
    bottom = cdf(d.untruncated, d.lower)
    return quantile(d.untruncated, bottom + p * (top - bottom))
end

median(d::Truncated) = quantile(d, 0.5)

function rand(d::Truncated)
    if d.nc > 0.25
        while true
            r = rand(d.untruncated)
            if d.lower <= r <= d.upper
                return r
            end
        end
    else
        return quantile(d.untruncated, cdf(d.untruncated, d.lower) + rand() * d.nc)
    end
end

# from fallbacks
function rand{D<:ContinuousUnivariateDistribution}(d::Truncated{D}, dims::Dims)
    return rand!(d, Array(Float64, dims))
end

function rand{D<:DiscreteUnivariateDistribution}(d::Truncated{D}, dims::Dims)
    return rand!(d, Array(Int, dims))
end
