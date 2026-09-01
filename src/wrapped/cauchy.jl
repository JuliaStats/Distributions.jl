function pdf(d::Wrapped{<:Cauchy}, x::Real; kwargs...)
    d0 = d.unwrapped
    l = d.lower
    u = d.upper
    period = u - l
    scale = d.k * 2π / period
    s = d0.σ * scale
    z = (x - d0.μ) * scale
    p = sinh(s) / (cosh(s) - cos(z)) / period
    l ≤ x < u || return zero(p)
    return p
end

function logpdf(d::Wrapped{<:Cauchy}, x::Real; kwargs...)
    d0 = d.unwrapped
    l = d.lower
    u = d.upper
    period = u - l
    scale = d.k * 2π / period
    s = d0.σ * scale
    z = (x - d0.μ) * scale
    logp = log1mexp(-2s) - logaddexp(2log1mexp(-s), logtwo - s + log1p(-cos(z)))
    l ≤ x < u || return oftype(logp, -Inf)
    return logp - log(oftype(logp, period))
end

function cdf(d::Wrapped{<:Cauchy}, x::Real; kwargs...)
    d0 = d.unwrapped
    l = d.lower
    u = d.upper
    k = d.k
    period = u - l
    scale = k * π / period
    s = d0.σ * scale
    z = (x - d0.μ) * scale
    zₗ = (l - d0.μ) * scale
    tanhs = tanh(s)
    p = (atan(tan(z), tanhs) - atan(tan(zₗ), tanhs)) / π / k
    x < l && return zero(p)
    x ≥ u && return one(p)
    # handle discontinuities in the antiderivative at tan((2i+1)/2 π) for integers i
    num_discontinuities = Int(div(z, π, RoundNearest)) - Int(div(zₗ, π, RoundNearest))
    return p + num_discontinuities // k
end

logcdf(d::Wrapped{<:Cauchy}, x::Real; kwargs...) = log(cdf(d, x; kwargs...))

