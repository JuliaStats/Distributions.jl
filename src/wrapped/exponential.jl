function pdf(d::Wrapped{<:Exponential}, x::Real; kwargs...)
    d0 = d.unwrapped
    l = d.lower
    u = d.upper
    period = u - l
    k = d.k
    scale = k / period
    z = scale * x
    λ = rate(d0)
    s = λ / scale
    p = λ * exp(-s * mod(z, 1)) / (1 - exp(-s)) / k
    l ≤ x < u && return p
    return zero(p)
end

function logpdf(d::Wrapped{<:Exponential}, x::Real; kwargs...)
    d0 = d.unwrapped
    l = d.lower
    u = d.upper
    period = u - l
    k = d.k
    scale = k / period
    z = scale * x
    λ = rate(d0)
    s = λ / scale
    p = log(λ) - s * mod(z, 1) - log1mexp(-s) - log(k)
    l ≤ x < u && return p
    return oftype(p, -Inf)
end

function cdf(d::Wrapped{<:Exponential}, x::Real; kwargs...)
    d0 = d.unwrapped
    l = d.lower
    u = d.upper
    period = u - l
    k = d.k
    scale = k / period
    z = scale * x
    zₗ = scale * l
    λ = rate(d0)
    s = λ / scale
    v = exp(-s)
    p = (1 + v^mod(zₗ, 1) - v^isinteger(zₗ) - v^mod(z, 1)) / (1 - v) / k
    x < l && return zero(p)
    x ≥ u && return one(p)
    # handle discontinuities in the antiderivative
    num_discontinuities = floor(Int, z) - floor(Int, zₗ) - isinteger(zₗ)
    return p + (num_discontinuities // k)
end

logcdf(d::Wrapped{<:Exponential}, x::Real; kwargs...) = log(cdf(d, x; kwargs...))
