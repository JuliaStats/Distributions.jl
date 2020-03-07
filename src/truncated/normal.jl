# Truncated normal distribution
"""
    TruncatedNormal(mu, sigma, l, u)

The *truncated normal distribution* is a particularly important one in the family of truncated distributions.
We provide additional support for this type with `TruncatedNormal` which calls `Truncated(Normal(mu, sigma), l, u)`.
Unlike the general case, truncated normal distributions support `mean`, `mode`, `modes`, `var`, `std`, and `entropy`.
"""
TruncatedNormal

@deprecate TruncatedNormal(mu::Real, sigma::Real, a::Real, b::Real) truncated(Normal(mu, sigma), a, b)

### statistics

Base.eltype(::Type{<:Truncated{Normal{T},Continuous}}) where {T <: Real} = T

minimum(d::Truncated{Normal{T},Continuous}) where {T <: Real} = d.lower
maximum(d::Truncated{Normal{T},Continuous}) where {T <: Real} = d.upper


function mode(d::Truncated{Normal{T},Continuous}) where T <: Real
    μ = mean(d.untruncated)
    d.upper < μ ? d.upper :
    d.lower > μ ? d.lower : μ
end

modes(d::Truncated{Normal{T},Continuous}) where {T <: Real} = [mode(d)]

# do not export. Used in mean, var
function _F1(x::Real, y::Real; thresh=1e-7)
    @assert 0 < thresh < Inf
    -Inf < x < Inf && -Inf < y < Inf || throw(DomainError())
    ϵ = exp(x^2 - y^2)
    if abs(x) > abs(y)
        _F1(y,x)
    elseif abs(x - y) ≤ thresh
        δ = y - x
        √π*x + (√π/2 + (-√π*x/6 + (-√π/12 + x*(√π/90 + (√π*x^2)/90)δ)δ)δ)δ
    elseif max(x,y) < 0
        (1 - ϵ) / (ϵ * erfcx(-y) - erfcx(-x))
    elseif min(x,y) > 0
        (1 - ϵ) / (erfcx(x) - ϵ * erfcx(y))
    else
        exp(-x^2) * (1 - ϵ) / (erf(y) - erf(x))
    end
end

# do not export. Used in mean, var
function _F2(x::Real, y::Real; thresh=1e-7)
    @assert 0 < thresh < Inf
    -Inf < x < Inf && -Inf < y < Inf || throw(DomainError())
    ϵ = exp(x^2 - y^2)
    if abs(x) > abs(y)
        _F2(y,x)
    elseif abs(x - y) ≤ thresh
        δ = y - x
        √π*x^2 - √π/2 + (√π*x + (√π/3 - √π*x^2/3 + (((√π/30 + √π*x^2/45)x^2 - 4*√π/45)δ - √π*x/3)δ)δ)δ
    elseif max(x,y) < 0
        (x - ϵ * y) / (ϵ * erfcx(-y) - erfcx(-x))
    elseif min(x,y) > 0
        (x - ϵ * y) / (erfcx(x) - ϵ * erfcx(y))
    else
        exp(-x^2) * (x - ϵ * y) / (erf(y) - erf(x))
    end
end

# do not export. Used in mean
function _tnmean(a, b)
    -Inf < a ≤ b < Inf || throw(DomainError())
    √(2/π) * _F1(a/√2, b/√2)
end

# do not export. Used in mean
function _tnmean(a, b, μ, σ)
    -Inf < a ≤ b < Inf || throw(DomainError())
    -Inf < μ < Inf && 0 < σ < Inf || throw(DomainError())
    α = (a - μ)/σ; β = (b - μ)/σ
    μ + _tnmean(α, β) * σ
end

# do not export. Used in var
function _tnvar(a, b)
    -Inf < a ≤ b < Inf || throw(DomainError())
    1 + 2/√π * _F2(a/√2, b/√2) - 2/π * _F1(a/√2, b/√2)^2
end

# do not export. Used in var
function _tnvar(a, b, μ, σ)
    -Inf < a ≤ b < Inf || throw(DomainError())
    -Inf < μ < Inf && 0 < σ < Inf || throw(DomainError())
    α = (a - μ)/σ; β = (b - μ)/σ
    _tnvar(α, β) * σ^2
end

function mean(d::Truncated{Normal{T},Continuous}) where T <: Real
    d0 = d.untruncated
    μ = mean(d0)
    σ = std(d0)
    if isfinite(d.lower) && isfinite(d.upper) && isfinite(μ) && isfinite(σ)
        # avoids loss of significance when truncation is far from μ.
        # See https://github.com/cossio/TruncatedNormal.jl/blob/master/notes/normal.pdf.
        _tnmean(d.lower, d.upper, μ, σ)
    else
        a = (d.lower - μ) / σ
        b = (d.upper - μ) / σ
        μ + ((normpdf(a) - normpdf(b)) / d.tp) * σ
    end
end

function var(d::Truncated{Normal{T},Continuous}) where T <: Real
    d0 = d.untruncated
    μ = mean(d0)
    σ = std(d0)
    if isfinite(d.lower) && isfinite(d.upper) && isfinite(μ) && isfinite(σ)
        # avoids loss of significance when truncation is far from μ.
        # See https://github.com/cossio/TruncatedNormal.jl/blob/master/notes/normal.pdf.
        _tnvar(d.lower, d.upper, μ, σ)
    else
        a = (d.lower - μ) / σ
        b = (d.upper - μ) / σ
        z = d.tp
        φa = normpdf(a)
        φb = normpdf(b)
        aφa = isinf(a) ? 0.0 : a * φa
        bφb = isinf(b) ? 0.0 : b * φb
        t1 = (aφa - bφb) / z
        t2 = abs2((φa - φb) / z)
        abs2(σ) * (1 + t1 - t2)
    end
end

function entropy(d::Truncated{Normal{T},Continuous}) where T <: Real
    d0 = d.untruncated
    z = d.tp
    μ = mean(d0)
    σ = std(d0)
    a = (d.lower - μ) / σ
    b = (d.upper - μ) / σ
    aφa = isinf(a) ? 0.0 : a * normpdf(a)
    bφb = isinf(b) ? 0.0 : b * normpdf(b)
    0.5 * (log2π + 1.) + log(σ * z) + (aφa - bφb) / (2.0 * z)
end


### sampling

## Use specialized sampler, as quantile-based method is inaccurate in
## tail regions of the Normal, issue #343

function rand(rng::AbstractRNG, d::Truncated{Normal{T},Continuous}) where T <: Real
    d0 = d.untruncated
    μ = mean(d0)
    σ = std(d0)
    a = (d.lower - μ) / σ
    b = (d.upper - μ) / σ
    z = randnt(rng, a, b, d.tp)
    return μ + σ * z
end

# Rejection sampler based on algorithm from Robert (1995)
#
#  - Available at http://arxiv.org/abs/0907.4010

function randnt(rng::AbstractRNG, lb::T, ub::T, tp::T) where {T<:AbstractFloat}
    if 3tp > 1   # has considerable chance of falling in [lb, ub]
        while true
            r = randn(rng, T)
            if lb ≤ r ≤ ub
                return r
            end
        end
    end
    span = ub - lb
    a = (sqrt(lb^2 + 4) + lb) / 2
    if lb > 0 && span > exp(lb * (lb - sqrt(lb^2 + 4)) / 4) / a
        while true
            r = rand(rng, Exponential(1 / a)) + lb
            u = rand(rng, T)
            if u < exp(-(r - a)^2 / 2) && r < ub
                return r
            end
        end
    end
    b = (sqrt(ub^2 + 4) - ub) / 2
    if ub < 0 && span > exp(ub * (ub + sqrt(ub^2 + 4)) / 4) / b
        while true
            r = rand(rng, Exponential(1 / b)) - ub
            u = rand(rng, T)
            if u < exp(-(r - b)^2 / 2) && r < -lb
                return -r
            end
        end
    end
    while true
        r = lb + rand(rng, T) * span
        u = rand(rng, T)
        if lb > 0
            rho = exp((lb^2 - r^2) / 2)
        elseif ub < 0
            rho = exp((ub^2 - r^2) / 2)
        else
            rho = exp(-r^2 / 2)
        end
        if u < rho
            return r
        end
    end
end
