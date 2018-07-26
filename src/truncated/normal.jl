# Truncated normal distribution
"""
    TruncatedNormal(mu, sigma, l, u)

The *truncated normal distribution* is a particularly important one in the family of truncated distributions.
We provide additional support for this type with `TruncatedNormal` which calls `Truncated(Normal(mu, sigma), l, u)`.
Unlike the general case, truncated normal distributions support `mean`, `mode`, `modes`, `var`, `std`, and `entropy`.
"""
TruncatedNormal(mu::Float64, sigma::Float64, a::Float64, b::Float64) =
    Truncated(Normal(mu, sigma), a, b)

TruncatedNormal(mu::Real, sigma::Real, a::Real, b::Real) =
    TruncatedNormal(Float64(mu), Float64(sigma), Float64(a), Float64(b))

### statistics

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

function rand(d::Truncated{Normal{T},Continuous}) where T <: Real
    d0 = d.untruncated
    μ = mean(d0)
    σ = std(d0)
    a = (d.lower - μ) / σ
    b = (d.upper - μ) / σ
    z = randnt(a, b, d.tp)
    return μ + σ * z
end

# Rejection sampler based on algorithm from Robert (1995)
#
#  - Available at http://arxiv.org/abs/0907.4010

function randnt(lb::Float64, ub::Float64, tp::Float64)
    local r::Float64
    if tp > 0.3   # has considerable chance of falling in [lb, ub]
        r = randn()
        while r < lb || r > ub
            r = randn()
        end
        return r

    else
        span = ub - lb
        if lb > 0 && span > 2.0 / (lb + sqrt(lb^2 + 4.0)) * exp((lb^2 - lb * sqrt(lb^2 + 4.0)) / 4.0)
            a = (lb + sqrt(lb^2 + 4.0))/2.0
            while true
                r = rand(Exponential(1.0 / a)) + lb
                u = rand()
                if u < exp(-0.5 * (r - a)^2) && r < ub
                    return r
                end
            end
        elseif ub < 0 && ub - lb > 2.0 / (-ub + sqrt(ub^2 + 4.0)) * exp((ub^2 + ub * sqrt(ub^2 + 4.0)) / 4.0)
            a = (-ub + sqrt(ub^2 + 4.0)) / 2.0
            while true
                r = rand(Exponential(1.0 / a)) - ub
                u = rand()
                if u < exp(-0.5 * (r - a)^2) && r < -lb
                    return -r
                end
            end
        else
            while true
                r = lb + rand() * (ub - lb)
                u = rand()
                if lb > 0
                    rho = exp((lb^2 - r^2) * 0.5)
                elseif ub < 0
                    rho = exp((ub^2 - r^2) * 0.5)
                else
                    rho = exp(-r^2 * 0.5)
                end
                if u < rho
                    return r
                end
            end
        end
    end
end
