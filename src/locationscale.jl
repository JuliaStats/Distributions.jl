immutable UnivariateLocationScaleFamily{T <: UnivariateDistribution} <: ContinuousUnivariateDistribution
    d::T
    location::Float64
    scale::Float64
    function UnivariateLocationScaleFamily(d::T, l::Real, s::Real)
        if s <= 0.0
            throw(ArgumentError("Scale must be non-negative"))
        end
        new(d, float64(l), float64(s))
    end
end

function UnivariateLocationScaleFamily{T <: UnivariateDistribution}(d::T, l::Real = 0.0, s::Real = 1.0)
    UnivariateLocationScaleFamily{T}(d, l, s)
end

function cdf(d::UnivariateLocationScaleFamily, x::Real)
    cdf(d.d, (x - d.location) / d.scale)
end

cf(d::UnivariateLocationScaleFamily) = error("Not yet implemented")

entropy(d::UnivariateLocationScaleFamily) = entropy(d.d)

function insupport(d::UnivariateLocationScaleFamily, x::Real)
    insupport(d.d, (x - d.location) / d.scale)
end

kurtosis(d::UnivariateLocationScaleFamily) = error("Not yet implemented")

location(d::UnivariateLocationScaleFamily) = d.location

function mean(d::UnivariateLocationScaleFamily)
    mean(d.d) * d.scale + d.location
end

function median(d::UnivariateLocationScaleFamily)
    median(d.d) * d.scale + d.location
end

mgf(d::UnivariateLocationScaleFamily) = error("Not yet implemented")

function modes(d::UnivariateLocationScaleFamily)
    m = modes(d.d)
    for i in 1:length(m)
        m[i] = m[i] * d.scale + d.location
    end
    return m
end

function pdf(d::UnivariateLocationScaleFamily, x::Real)
    1 / d.scale * pdf(d.d, (x - d.location) / d.scale)
end

function quantile(d::UnivariateLocationScaleFamily, p::Real)
    quantile(d.d, p) * d.scale + d.location
end

scale(d::UnivariateLocationScaleFamily) = d.scale

skewness(d::UnivariateLocationScaleFamily) = error("Not yet implemented")

var(d::UnivariateLocationScaleFamily) = var(d.d) * d.scale^2

function rand(d::UnivariateLocationScaleFamily)
    rand(d.d) * d.scale + d.location
end

# Fallbacks
location(d::ContinuousUnivariateDistribution) = 0.0
scale(d::ContinuousUnivariateDistribution) = 1.0

location(d::DiscreteUnivariateDistribution) = 0
scale(d::DiscreteUnivariateDistribution) = 1

# Relocate
function Base.(:+)(d::UnivariateDistribution, a::Real)
    UnivariateLocationScaleFamily(d, a)
end

function Base.(:+)(a::Real, d::UnivariateDistribution)
    UnivariateLocationScaleFamily(d, a)
end

function Base.(:+)(d::UnivariateLocationScaleFamily, a::Real)
    UnivariateLocationScaleFamily(d.d, location(d) + a, scale(d))
end

function Base.(:+)(a::Real, d::UnivariateLocationScaleFamily)
    UnivariateLocationScaleFamily(d.d, location(d) + a, scale(d))
end

# Rescale
function Base.(:*)(d::UnivariateDistribution, b::Real)
    UnivariateLocationScaleFamily(d, 0.0, b)
end

function Base.(:*)(b::Real, d::UnivariateDistribution)
    UnivariateLocationScaleFamily(d, 0.0, b)
end

function Base.(:*)(d::UnivariateLocationScaleFamily, b::Real)
    UnivariateLocationScaleFamily(d.d, location(d), scale(d) * b)
end

function Base.(:*)(b::Real, d::UnivariateLocationScaleFamily)
    UnivariateLocationScaleFamily(d.d, location(d), scale(d) * b)
end
