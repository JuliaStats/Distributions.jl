### statistics

function mode(d::Truncated{<:Normal{<:Real},Continuous,T}) where {T<:Real}
    μ = mean(d.untruncated)
    return T(clamp(μ, extrema(d)...))
end

modes(d::Truncated{<:Normal{<:Real},Continuous}) = [mode(d)]

# do not export. Used in mean
# computes mean of standard normal distribution truncated to [a, b]
function _tnmom1(a, b)
    mid = float(middle(a, b))
    if !(a ≤ b)
        return oftype(mid, NaN)
    elseif a == b
        return mid
    elseif abs(a) > abs(b)
        return -_tnmom1(-b, -a)
    elseif isinf(a) && isinf(b)
        return zero(mid)
    end
    Δ = (b - a) * mid
    a′ = a * invsqrt2
    b′ = b * invsqrt2
    if a ≤ 0 ≤ b
        m = expm1(-Δ) * exp(-a^2 / 2) / erf(b′, a′)
    elseif 0 < a < b
        z = exp(-Δ) * erfcx(b′) - erfcx(a′)
        iszero(z) && return mid
        m = expm1(-Δ) / z
    end
    return clamp(m / sqrthalfπ, a, b)
end

# do not export. Used in var
# computes 2nd moment of standard normal distribution truncated to [a, b]
function _tnmom2(a::Real, b::Real)
    mid = float(middle(a, b))
    if !(a ≤ b)
        return oftype(mid, NaN)
    elseif a == b
        return mid^2
    elseif abs(a) > abs(b)
        return _tnmom2(-b, -a)
    elseif isinf(a) && isinf(b)
        return one(mid)
    elseif isinf(b)
        return 1 + a / erfcx(a * invsqrt2) / sqrthalfπ
    end
    a′ = a * invsqrt2
    b′ = b * invsqrt2
    if a ≤ 0 ≤ b
        eb_ea = sqrthalfπ * erf(a′, b′)
        fb_fa = eb_ea + a * exp(-a^2 / 2) - b * exp(-b^2 / 2)
        return fb_fa / eb_ea
    else # 0 ≤ a ≤ b
        exΔ = exp((a - b) * mid)
        ea = sqrthalfπ * erfcx(a′)
        eb = sqrthalfπ * erfcx(b′)
        fa = ea + a
        fb = eb + b
        return (fa - fb * exΔ) / (ea - eb * exΔ)
    end
end

# do not export. Used in var
function _tnvar(a::Real, b::Real)
    if a == b
        return zero(middle(a, b))
    elseif a < b
        m1 = _tnmom1(a, b)
        m2 = √_tnmom2(a, b)
        return (m2 - m1) * (m2 + m1)
    else
        return oftype(middle(a, b), NaN)
    end
end

function mean(d::Truncated{<:Normal{<:Real},Continuous,T}) where {T<:Real}
    d0 = d.untruncated
    μ = mean(d0)
    σ = std(d0)
    if iszero(σ)
        return mode(d)
    else
        lower, upper = extrema(d)
        a = (lower - μ) / σ
        b = (upper - μ) / σ
        return T(μ + _tnmom1(a, b) * σ)
    end
end

function var(d::Truncated{<:Normal{<:Real},Continuous,T}) where {T<:Real}
    d0 = d.untruncated
    μ = mean(d0)
    σ = std(d0)
    if iszero(σ)
        return T(σ)
    else
        lower, upper = extrema(d)
        a = (lower - μ) / σ
        b = (upper - μ) / σ
        return T(_tnvar(a, b) * σ^2)
    end
end

function entropy(d::Truncated{<:Normal{<:Real},Continuous})
    d0 = d.untruncated
    z = d.tp
    μ = mean(d0)
    σ = std(d0)
    lower, upper = extrema(d)
    a = (lower - μ) / σ
    b = (upper - μ) / σ
    aφa = isinf(a) ? 0.0 : a * normpdf(a)
    bφb = isinf(b) ? 0.0 : b * normpdf(b)
    0.5 * (log2π + 1.) + log(σ * z) + (aφa - bφb) / (2.0 * z)
end


### sampling

## Use specialized sampler, as quantile-based method is inaccurate in
## tail regions of the Normal, issue #343

function rand(rng::AbstractRNG, d::Truncated{<:Normal{<:Real},Continuous})
    d0 = d.untruncated
    μ = mean(d0)
    σ = std(d0)
    if isfinite(μ)
        lower, upper = extrema(d)
        a = (lower - μ) / σ
        b = (upper - μ) / σ
        z = randnt(rng, a, b, d.tp)
        return μ + σ * z
    else
        return clamp(μ, extrema(d)...)
    end
end

# Rejection sampler based on algorithm from Robert (1995)
#
#  - Available at http://arxiv.org/abs/0907.4010

function randnt(rng::AbstractRNG, lb::Float64, ub::Float64, tp::Float64)
    local r::Float64
    if tp > 0.3   # has considerable chance of falling in [lb, ub]
        r = randn(rng)
        while r < lb || r > ub
            r = randn(rng)
        end
        return r

    else
        span = ub - lb
        if lb > 0 && span > 2.0 / (lb + sqrt(lb^2 + 4.0)) * exp((lb^2 - lb * sqrt(lb^2 + 4.0)) / 4.0)
            a = (lb + sqrt(lb^2 + 4.0))/2.0
            while true
                r = rand(rng, Exponential(1.0 / a)) + lb
                u = rand(rng)
                if u < exp(-0.5 * (r - a)^2) && r < ub
                    return r
                end
            end
        elseif ub < 0 && ub - lb > 2.0 / (-ub + sqrt(ub^2 + 4.0)) * exp((ub^2 + ub * sqrt(ub^2 + 4.0)) / 4.0)
            a = (-ub + sqrt(ub^2 + 4.0)) / 2.0
            while true
                r = rand(rng, Exponential(1.0 / a)) - ub
                u = rand(rng)
                if u < exp(-0.5 * (r - a)^2) && r < -lb
                    return -r
                end
            end
        else
            while true
                r = lb + rand(rng) * (ub - lb)
                u = rand(rng)
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
