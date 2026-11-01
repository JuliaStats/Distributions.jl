#####
##### Truncated log-normal distribution
#####

# Mean and variance of `truncated(LogNormal(μ, σ); lower=l)` and `truncated(LogNormal(μ, σ); upper=u)`.
#
# Let X = exp(μ + σZ) with Z ~ Normal(0, 1), and let M(z) = Q(z) / φ(z) = sqrthalfπ * erfcx(z / √2)
# be the Mills ratio of Z, with Q and φ its complementary cdf and pdf.
#
# For a lower bound l > 0 and a = (log(l) - μ) / σ, the moments are
#     E[X^k | X ≥ l] = exp(kμ + k²σ²/2) Q(a - kσ) / Q(a) = l^k M(a - kσ) / M(a).
# The variance is computed as
#     var = mean² * expm1(D),    D = h(a - 2σ) - 2h(a - σ) + h(a),    h(z) = log(M(z)).
# For small σ or large a this second difference cancels catastrophically (as does the difference
# of the second moment and the squared mean). Since h''(z) = V(z), the variance of Z truncated to
# [z, ∞) (`_tnvar_lower`), the second difference is the integral of a positive function:
#     D = σ² ∫₀¹ (1 - s) (V(c + σs) + V(c - σs)) ds,    c = a - σ,
# which Gauss-Legendre quadrature evaluates without cancellation (`_logmills_d2`).
#
# An upper bound u > 0 with b = (log(u) - μ) / σ mirrors the lower bound: since Z ≤ b if and only
# if -Z ≥ -b, E[X^k | X ≤ u] = u^k M(kσ - b) / M(-b), and D is the same second difference centred
# at c = σ - b.

# Variance of the standard normal distribution truncated to [z, ∞) (for finite z).
function _tnvar_lower(z::Real)
    if !(z ≥ 2)
        # λ = φ(z) / Q(z) is the inverse Mills ratio. For z < 2 the cancellation in
        # 1 - λ (λ - z) costs at most about two decimal digits. Below zero, Q(z) ≥ 1/2 and
        # φ(z) underflows gracefully, whereas erfcx overflows for z < -37.7 (an infinite
        # intermediate whose derivative turns reverse-mode gradients into NaN).
        λ = if z < 0
            2 * invsqrt2π * exp(-z^2 / 2) / erfc(z * invsqrt2)
        else
            inv(sqrthalfπ * erfcx(z * invsqrt2))
        end
        return 1 - λ * (λ - z)
    else
        # Laplace's continued fraction λ - z = t₁ with t_k = k / (z + t_{k+1}), evaluated backwards
        # from the fixed point of t = (n + 1) / (z + t). The number of terms gives full double
        # precision for z ≥ 2. Then V = 1 - z t₁ - t₁² = t₁ (t₂ - t₁) without cancellation.
        n = ceil(Int, 6 + 120 / z + 100 / z^2)
        t = 2 * (n + 1) / (sqrt(z^2 + 4 * (n + 1)) + z)
        for k in n:-1:2
            t = k / (z + t)
        end
        t₁ = inv(z + t)
        return t₁ * (t - t₁)
    end
end

# Nodes s_i of the 8-point Gauss-Legendre rule on [0, 1], and its weights multiplied by (1 - s_i)
const _LOGMILLS_D2_NODES = (
    0.019855071751231884, 0.10166676129318664, 0.2372337950418355, 0.4082826787521751,
    0.591717321247825, 0.7627662049581645, 0.8983332387068134, 0.9801449282487681,
)
const _LOGMILLS_D2_WEIGHTS = (
    0.04960931821952933, 0.09988613745373567, 0.11964241387321545, 0.10730313838033538,
    0.07403875330884561, 0.037210909065728194, 0.011304379772951567, 0.0010049499256588007,
)

# Whether `_logmills_d2(c, σ)` uses quadrature: the rule is accurate to double precision while
# σ is at most about a third of the distance from c to the nearest complex zero of the Mills
# ratio (near -1.92 ± 2.82i). Beyond that, D is large enough for the direct second difference.
# The quadrature is tuned to double precision; in higher precision the direct second difference
# keeps enough digits.
_logmills_d2_quadrature(c::Real, σ::Real) = σ ≤ 0.35 * hypot(c + 2, 3)
_logmills_d2_quadrature(c::BigFloat, σ::BigFloat) = false

# Second difference h(c - σ) - 2h(c) + h(c + σ) of the logarithm h(z) = log(Q(z) / φ(z)) of the
# Mills ratio of the standard normal distribution (for finite c and σ > 0).
function _logmills_d2(c::Real, σ::Real)
    if _logmills_d2_quadrature(c, σ)
        # h'' = V, so the second difference is σ² ∫₀¹ (1 - s) (V(c + σs) + V(c - σs)) ds
        s = zero(float(promote_type(typeof(c), typeof(σ))))
        for (x, w) in zip(_LOGMILLS_D2_NODES, _LOGMILLS_D2_WEIGHTS)
            s += w * (_tnvar_lower(c + σ * x) + _tnvar_lower(c - σ * x))
        end
        return σ^2 * s
    elseif c + σ ≤ 0
        # h(z) = z²/2 + log(Q(z)) + log(√(2π)), and the second difference of z²/2 is σ²
        return σ^2 + ((normlogccdf(c - σ) - 2 * normlogccdf(c)) + normlogccdf(c + σ))
    else
        # h(z) = log(erfcx(z / √2)) + log(sqrthalfπ)
        return (logerfcx((c - σ) * invsqrt2) - 2 * logerfcx(c * invsqrt2)) + logerfcx((c + σ) * invsqrt2)
    end
end

# Mean of `truncated(LogNormal(μ, σ); lower=l)` for l > 0 and σ > 0, and the centre c = a - σ of the
# second difference that gives its variance.
function _tlognormal_lower_mean(μ::Real, σ::Real, l::Real)
    a = (log(l) - μ) / σ
    m = if a ≥ σ
        # l M(a - σ) / M(a), with both arguments of erfcx nonnegative
        l * (erfcx((a - σ) * invsqrt2) / erfcx(a * invsqrt2))
    else
        exp(μ + σ^2 / 2 + (normlogccdf(a - σ) - normlogccdf(a)))
    end
    return m, a - σ
end

# Mean of `truncated(LogNormal(μ, σ); upper=u)` for u > 0 and σ > 0, and the centre c = σ - b of the
# second difference that gives its variance.
function _tlognormal_upper_mean(μ::Real, σ::Real, u::Real)
    b = (log(u) - μ) / σ
    m = if b ≤ 0
        # u M(σ - b) / M(-b), with both arguments of erfcx nonnegative
        u * (erfcx((σ - b) * invsqrt2) / erfcx(-b * invsqrt2))
    else
        exp(μ + σ^2 / 2 + (normlogcdf(b - σ) - normlogcdf(b)))
    end
    return m, σ - b
end

### statistics

function mean(d::LeftTruncated{<:LogNormal,Continuous,T}) where {T<:Real}
    d0 = d.untruncated
    μ, σ = params(d0)
    l = d.lower
    if !(l > 0)
        # the bound is not in the interior of the support
        return T(mean(d0))
    elseif iszero(σ)
        return T(clamp(exp(μ), extrema(d)...))
    end
    m, _ = _tlognormal_lower_mean(μ, σ, l)
    return T(m)
end

function var(d::LeftTruncated{<:LogNormal,Continuous,T}) where {T<:Real}
    d0 = d.untruncated
    μ, σ = params(d0)
    l = d.lower
    if !(l > 0)
        # the bound is not in the interior of the support
        return T(var(d0))
    elseif iszero(σ)
        return zero(T)
    end
    m, c = _tlognormal_lower_mean(μ, σ, l)
    return T(m^2 * expm1(_logmills_d2(c, σ)))
end

function mean(d::RightTruncated{<:LogNormal,Continuous,T}) where {T<:Real}
    d0 = d.untruncated
    μ, σ = params(d0)
    u = d.upper
    if !(u > 0)
        # the truncated distribution has no mass
        return T(NaN)
    elseif isinf(u)
        return T(mean(d0))
    elseif iszero(σ)
        return T(clamp(exp(μ), extrema(d)...))
    end
    m, _ = _tlognormal_upper_mean(μ, σ, u)
    return T(m)
end

function var(d::RightTruncated{<:LogNormal,Continuous,T}) where {T<:Real}
    d0 = d.untruncated
    μ, σ = params(d0)
    u = d.upper
    if !(u > 0)
        # the truncated distribution has no mass
        return T(NaN)
    elseif isinf(u)
        return T(var(d0))
    elseif iszero(σ)
        return zero(T)
    end
    m, c = _tlognormal_upper_mean(μ, σ, u)
    return T(m^2 * expm1(_logmills_d2(c, σ)))
end
