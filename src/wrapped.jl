using Distributions

"""
    wrapped(d0::UnivariateDistribution, l::Real, u::Real; k::Int=1, kwargs...)

Construct a ``k``-wrapped distribution with support ``[l, u)``.

The resulting density wraps around the interval. With the period ``\\tau = u - l``, the
density can be expressed as two infinite series in terms of either the density ``f_0`` of
the unwrapped distribution `d0` or in terms of its characteristic function [`cf`](@ref)
``\\phi_0``:
````math
\\begin{aligned}
f(x) &= \\frac{1}{\\tau}\\sum_{j=-\\infty}^\\infty f_0\\left(x + \\frac{j}{k}\\tau\\right)\\
     &= \\frac{1}{\\tau}\\sum_{j=-\\infty}^\\infty
            \\phi_0\\left(jk\\frac{2\\pi}{\\tau}\\right)
            \\exp\\left(-ijk\\frac{2\\pi}{\\tau}x\\right)
\\end{aligned}
````
If `k>1`, then the resulting distribution has ``k``-fold symmetry in the interval.

# Keywords
- `characteristic::Bool`: whether to use the characteristic function to compute the density.
    If `false`, the density is computed using the logpdf of the unwrapped distribution.
    If not specified, the `wrapped` method called may choose based on the parameters of
    `l`, `u`, `tol`, and `d0`.
- `tol`: target absolute error of `pdf`. May be used to determine whether to use the
    characteristic function in computations or not.

# Examples

"""
function wrapped(d::UnivariateDistribution, l::Real, u::Real; kwargs...)
    return Wrapped(d, promote(l, u)...; kwargs...)
end
function wrapped(d::UnivariateDistribution, l::T, u::T; kwargs...) where {T<:Real}
    return Wrapped(d, l, u; kwargs...)
end

"""
    Wrapped

Generic wrapper for a wrapped distribution.
"""
struct Wrapped{D<:UnivariateDistribution,S<:ValueSupport,T<:Real} <:
       UnivariateDistribution{S}
    unwrapped::D          # the original distribution (untruncated)
    lower::T              # lower bound
    upper::T              # upper bound
    characteristic::Bool  # whether to use characteristic function
    k::Int                # k-fold symmetry
    function Wrapped(
        d::UnivariateDistribution,
        l::T,
        u::T;
        tol::Real=sqrt(eps(float(T))),
        characteristic::Bool=_wrapped_use_characteristic(d, l, u, tol),
        k::Int=1,
    ) where {T<:Real}
        return new{typeof(d),value_support(typeof(d)),T}(
            d, l, u, characteristic, k
        )
    end
end

_wrapped_use_characteristic(d::UnivariateDistribution, l, u, tol) = false

minimum(d::Wrapped) = d.lower
maximum(d::Wrapped) = d.upper - eps(float(d.upper))

function logpdf(
    d::Wrapped, x::Real; characteristic::Bool=d.characteristic, kwargs...
)
    if characteristic
        log(pdf_wrapped_cf(d, x; kwargs...))
    else
        logpdf_wrapped_logpdf(d, x; kwargs...)
    end
end

logpdf_wrapped_cf(d::Wrapped, x::Real; kwargs...) = log(pdf_wrapped_cf(d, x; kwargs...))

function logpdf_wrapped_logpdf(
    d::Wrapped, x::Real; tol=sqrt(eps(float(typeof(x)))), maxiter=1_000
)
    d_unwrapped = d.unwrapped
    period = d.upper - d.lower
    k = d.k
    logp = logpdf(d_unwrapped, x)
    insupport(d, x) || return oftype(logp, -Inf)
    Δx = period / k
    x₋ = x - Δx
    x₊ = x + Δx
    x₋_in_support = insupport(d_unwrapped, x₋)
    x₊_in_support = insupport(d_unwrapped, x₊)
    for i in 1:maxiter
        Δlogp = oftype(logp, -Inf)
        if x₋_in_support
            Δlogp = logaddexp(Δlogp, logpdf(d_unwrapped, x₋))
            x₋ -= Δx
            x₋_in_support = insupport(d_unwrapped, x₋)
        end
        if x₊_in_support
            Δlogp = logaddexp(Δlogp, logpdf(d_unwrapped, x₊))
            x₊ += Δx
            x₊_in_support = insupport(d_unwrapped, x₊)
        end
        logp = logaddexp(logp, Δlogp)
        i > k && abs(Δlogp) ≤ tol && isfinite(logp) && break
    end
    return logp - log(oftype(logp, k))
end

function pdf(
    d::Wrapped, x::Real; characteristic::Bool=d.characteristic, kwargs...
)
    if characteristic
        pdf_wrapped_cf(d, x; kwargs...)
    else
        pdf_wrapped_pdf(d, x; kwargs...)
    end
end

function pdf_wrapped_pdf(
    d::Wrapped, x::Real; tol=sqrt(eps(float(typeof(x)))), maxiter=1_000
)
    d_unwrapped = d.unwrapped
    period = d.upper - d.lower
    k = d.k
    p = pdf(d_unwrapped, x)
    insupport(d, x) || return zero(p)
    Δx = period / k
    x₋ = x - Δx
    x₊ = x + Δx
    x₋_in_support = insupport(d_unwrapped, x₋)
    x₊_in_support = insupport(d_unwrapped, x₊)
    for i in 1:maxiter
        Δp = zero(p)
        if x₋_in_support
            Δp += pdf(d_unwrapped, x₋)
            x₋ -= Δx
            x₋_in_support = insupport(d_unwrapped, x₋)
        end
        if x₊_in_support
            Δp += pdf(d_unwrapped, x₊)
            x₊ += Δx
            x₊_in_support = insupport(d_unwrapped, x₊)
        end
        p += Δp
        i > k && abs(Δp) ≤ tol && p > 0 && break
    end
    return p / k
end

function cdf(
    d::Wrapped, x::Real; characteristic::Bool=d.characteristic, kwargs...
)
    if characteristic && !(d.unwrapped isa DiscreteDistribution)
        cdf_wrapped_cf(d, x; kwargs...)
    else
        cdf_wrapped_cdf(d, x; kwargs...)
    end
end

function logcdf(
    d::Wrapped, x::Real; characteristic::Bool=d.characteristic, kwargs...
)
    if characteristic && !(d.unwrapped isa DiscreteDistribution)
        log(cdf_wrapped_cf(d, x; kwargs...))
    else
        logcdf_wrapped_logcdf(d, x; kwargs...)
    end
end

function cdf_wrapped_cdf(
    d::Wrapped, x::Real; tol=sqrt(eps(float(typeof(x)))), maxiter=1_000
)
    d_unwrapped = d.unwrapped
    l = d.lower
    period = d.upper - l
    k = d.k
    p = cdf(d_unwrapped, x) - cdf(d_unwrapped, l)
    x < l && return zero(p)
    Δx = period / k
    x₋ = x - Δx
    x₊ = x + Δx
    x₋_in_support = insupport(d_unwrapped, x₋)
    l₋ = l - Δx
    l₊ = l + Δx
    l₊_in_support = insupport(d_unwrapped, l₊)
    for i in 1:maxiter
        Δp = zero(p)
        if x₋_in_support
            Δp += cdf(d_unwrapped, x₋) - cdf(d_unwrapped, l₋)
            if d_unwrapped isa DiscreteDistribution
                Δp += pdf(d_unwrapped, l₋)
            end
            x₋ -= Δx
            l₋ -= Δx
            x₋_in_support = insupport(d_unwrapped, x₋)
        end
        if l₊_in_support
            Δp += cdf(d_unwrapped, x₊) - cdf(d_unwrapped, l₊)
            if d_unwrapped isa DiscreteDistribution
                Δp += pdf(d_unwrapped, l₊)
            end
            x₊ += Δx
            l₊ += Δx
            l₊_in_support = insupport(d_unwrapped, l₊)
        end
        p += Δp
        i > k && abs(Δp) ≤ tol && p > 0 && break
    end
    return p / k
end

function logcdf_wrapped_logcdf(
    d::Wrapped, x::Real; tol=sqrt(eps(float(typeof(x)))), maxiter=1_000
)
    d_unwrapped = d.unwrapped
    l = d.lower
    period = d.upper - l
    k = d.k
    x < l && return oftype(logcdf(d_unwrapped, l), -Inf)
    logp = logdiffcdf(d_unwrapped, x, l)
    x < l && return oftype(logp, -Inf)
    Δx = period / k
    x₋ = x - Δx
    x₊ = x + Δx
    x₋_in_support = insupport(d_unwrapped, x₋)
    l₋ = l - Δx
    l₊ = l + Δx
    l₊_in_support = insupport(d_unwrapped, l₊)
    for i in 1:maxiter
        Δlogp = oftype(logp, -Inf)
        if x₋_in_support
            Δlogp = logaddexp(Δlogp, logdiffcdf(d_unwrapped, x₋, l₋))
            if d_unwrapped isa DiscreteDistribution
                Δlogp = logaddexp(Δlogp, logpdf(d_unwrapped, l₋))
            end
            x₋ -= Δx
            l₋ -= Δx
            x₋_in_support = insupport(d_unwrapped, x₋)
        end
        if l₊_in_support
            Δlogp = logaddexp(Δlogp, logdiffcdf(d_unwrapped, x₊, l₊))
            if d_unwrapped isa DiscreteDistribution
                Δlogp = logaddexp(Δlogp, logpdf(d_unwrapped, l₊))
            end
            x₊ += Δx
            l₊ += Δx
            l₊_in_support = insupport(d_unwrapped, l₊)
        end
        logp = logaddexp(logp, Δlogp)
        i > k && abs(Δlogp) ≤ tol && isfinite(logp) && break
    end
    return min(logp - log(oftype(logp, k)), 0)
end

function pdf_wrapped_cf(d::Wrapped, x::Real; tol=sqrt(eps(float(typeof(x)))), maxiter=1_000)
    d_unwrapped = d.unwrapped
    period = d.upper - d.lower
    k = d.k
    scale = twoπ * (k / period)
    iscale = zero(scale)
    z₀ = cis(-x * scale)
    z = one(z₀)
    p = real(cf(d_unwrapped, iscale) * z)
    insupport(d, x) || return zero(p)
    for i in 1:maxiter
        z *= z₀
        iscale += scale
        Δp = 2 * real(cf(d_unwrapped, iscale) * z)
        p += Δp
        i > k && abs(Δp) ≤ tol && p > 0 && break
    end
    # ensure density is never negative
    return max(0, p) / period
end

function cdf_wrapped_cf(d::Wrapped, x::Real; tol=sqrt(eps(float(typeof(x)))), maxiter=1_000)
    d_unwrapped = d.unwrapped
    period = d.upper - d.lower
    k = d.k
    scale = twoπ * (k / period)
    iscale = zero(scale)
    z₀ = cis(-x * scale)
    z₀ₗ = cis(-d.lower * scale)
    z = zₗ = one(z₀)
    p = real(cf(d_unwrapped, iscale)) * (x - d.lower)
    for i in 1:maxiter
        z *= z₀
        zₗ *= z₀ₗ
        iscale += scale
        Δp = 2 * imag(cf(d_unwrapped, iscale) * (zₗ - z)) / iscale
        p += Δp
        i > k && abs(Δp) ≤ tol && p > 0 && break
    end
    # ensure probability is never negative
    return clamp(p / period, 0, 1)
end

function rand(rng::AbstractRNG, d::Wrapped)
    l = d.lower
    k = d.k
    period = d.upper - l
    # choose which of the `k` folds the point will be from
    i = k == 1 ? k : rand(rng, 1:k)
    z = rand(rng, d.unwrapped)
    return mod(z - l + ((k - i)//k) * period, period) + l
end

### specialized wrapped distributions

include(joinpath("wrapped", "normal.jl"))
include(joinpath("wrapped", "cauchy.jl"))
include(joinpath("wrapped", "exponential.jl"))
