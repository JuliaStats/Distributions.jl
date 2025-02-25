# Implementation based on chapters 2-4 of
# Arnold, Barry C., Narayanaswamy Balakrishnan, and Haikady Navada Nagaraja.
# A first course in order statistics. Society for Industrial and Applied Mathematics, 2008.

"""
    OrderStatistic{D<:UnivariateDistribution,S<:ValueSupport} <: UnivariateDistribution{S}

The distribution of an order statistic from IID samples from a univariate distribution.

    OrderStatistic(dist::UnivariateDistribution, n::Int, rank::Int; check_args::Bool=true)

Construct the distribution of the `rank` ``=i``th order statistic from `n` independent
samples from `dist`.

The ``i``th order statistic of a sample is the ``i``th element of the sorted sample.
For example, the 1st order statistic is the sample minimum, while the ``n``th order
statistic is the sample maximum.

If ``f`` is the probability density (mass) function of `dist` with distribution function
``F``, then the probability density function ``g`` of the order statistic for continuous
`dist` is
```math
g(x; n, i) = {n \\choose i} [F(x)]^{i-1} [1 - F(x)]^{n-i} f(x),
```
and the probability mass function ``g`` of the order statistic for discrete `dist` is
```math
g(x; n, i) = \\sum_{k=i}^n {n \\choose k} \\left( [F(x)]^k [1 - F(x)]^{n-k} - [F(x_-)]^k [1 - F(x_-)]^{n-k} \\right),
```
where ``x_-`` is the largest element in the support of `dist` less than ``x``.

For the joint distribution of a subset of order statistics, use
[`JointOrderStatistics`](@ref) instead.

## Examples

```julia
OrderStatistic(Cauchy(), 10, 1)              # distribution of the sample minimum
OrderStatistic(DiscreteUniform(10), 10, 10)  # distribution of the sample maximum
OrderStatistic(Gamma(1, 1), 11, 5)           # distribution of the sample median
```
"""
struct OrderStatistic{D<:UnivariateDistribution,S<:ValueSupport} <:
       UnivariateDistribution{S}
    dist::D
    n::Int
    rank::Int
    function OrderStatistic(
        dist::UnivariateDistribution, n::Int, rank::Int; check_args::Bool=true
    )
        @check_args(OrderStatistic, 1 ≤ rank ≤ n)
        return new{typeof(dist),value_support(typeof(dist))}(dist, n, rank)
    end
end

minimum(d::OrderStatistic) = minimum(d.dist)
maximum(d::OrderStatistic) = maximum(d.dist)
insupport(d::OrderStatistic, x::Real) = insupport(d.dist, x)

params(d::OrderStatistic) = tuple(params(d.dist)..., d.n, d.rank)
partype(d::OrderStatistic) = partype(d.dist)
Base.eltype(::Type{<:OrderStatistic{D}}) where {D} = Base.eltype(D)
Base.eltype(d::OrderStatistic) = eltype(d.dist)

# distribution of the ith order statistic from an IID uniform distribution, with CDF Uᵢₙ(x)
function _uniform_orderstatistic(d::OrderStatistic)
    n = d.n
    rank = d.rank
    return Beta{Int}(rank, n - rank + 1)
end

function logpdf(d::OrderStatistic, x::Real)
    b = _uniform_orderstatistic(d)
    p = cdf(d.dist, x)
    if value_support(typeof(d)) === Continuous
        return logpdf(b, p) + logpdf(d.dist, x)
    else
        return logdiffcdf(b, p, p - pdf(d.dist, x))
    end
end

for f in (:logcdf, :logccdf, :cdf, :ccdf)
    @eval begin
        function $f(d::OrderStatistic, x::Real)
            b = _uniform_orderstatistic(d)
            return $f(b, cdf(d.dist, x))
        end
    end
end

for f in (:quantile, :cquantile)
    @eval begin
        function $f(d::OrderStatistic, p::Real)
            # since cdf is Fᵢₙ(x) = Uᵢₙ(Fₓ(x)), and Uᵢₙ is invertible and increasing, we
            # have Fₓ(x) = Uᵢₙ⁻¹(Fᵢₙ(x)). then quantile function is
            # Qᵢₙ(p) = inf{x: p ≤ Fᵢₙ(x)} = inf{x: Uᵢₙ⁻¹(p) ≤ Fₓ(x)} = Qₓ(Uᵢₙ⁻¹(p))
            b = _uniform_orderstatistic(d)
            return quantile(d.dist, $f(b, p))
        end
    end
end

function rand(rng::AbstractRNG, d::OrderStatistic)
    # inverse transform sampling. Since quantile function is Qₓ(Uᵢₙ⁻¹(p)), we draw a random
    # variable from Uᵢₙ and pass it through the quantile function of `d.dist`
    T = eltype(d.dist)
    b = _uniform_orderstatistic(d)
    return T(quantile(d.dist, rand(rng, b)))
end

# Moments

## Fallbacks

mean(d::OrderStatistic) = _moment(d, 1)
var(d::OrderStatistic) = _moment(d, 2, _moment(d, 1))
function skewness(d::OrderStatistic)
    m = mean(d)
    return _moment(d, 3, m) / _moment(d, 2, m)^(3//2)
end
function kurtosis(d::OrderStatistic)
    m = mean(d)
    return _moment(d, 4, m) / _moment(d, 2, m)^2 - 3
end

function _moment(
    d::OrderStatistic{<:ContinuousUnivariateDistribution},
    p::Int,
    μ::Real=0;
    rtol=sqrt(eps(promote_type(partype(d), typeof(μ)))),
)
    # assumes if p == 1, then μ=0 and if p>1, then μ==mean(d)
    T = float(typeof(one(Base.promote_type(partype(d), typeof(μ)))))
    n = d.n
    rank = d.rank

    μdist = mean(d.dist)
    isfinite(μ) && isfinite(μdist) || return T(NaN)
    if isodd(p) && isodd(n) && rank == (n + 1) ÷ 2 && _issymmetric(d.dist)
        # for symmetric distributions, distribution of median is also symmetric, so all of
        # its odd central moments are 0.
        return p == 1 ? μdist : zero(T)
    end

    logC = -logbeta(rank, T(n - rank + 1))
    function integrand(x)
        if x < μ
            # for some distributions (e.g. Normal) this improves numerical stability
            logF = logcdf(d.dist, x)
            log1mF = log1mexp(logF)
        else
            log1mF = logccdf(d.dist, x)
            logF = log1mexp(log1mF)
        end
        return (x - μ)^p *
               exp(logC + logpdf(d.dist, x) + (rank - 1) * logF + (n - rank) * log1mF)
    end
    α = eps(T) / 2
    lower = quantile(d, α)
    upper = quantile(d, 1 - α)
    return first(quadgk(integrand, lower, upper; rtol=rtol))
end

function _moment(d::OrderStatistic{<:DiscreteUnivariateDistribution}, p::Int, μ::Real=0)
    T = float(typeof(one(Base.promote_type(partype(d), typeof(μ)))))

    if isbounded(d.dist)
        xs = support(d.dist)
    elseif eltype(d.dist) <: Integer
        # Note: this approximation can fail in the unlikely case of an atom with huge
        # magnitude just outside of the bulk.
        α = eps(T) / 2
        xmin = quantile(d, α)
        xmax = quantile(d, 1 - α)
        xs = xmin:xmax
    else
        throw(
            ArgumentError(
                "Moments can only be computed for bounded distributions or those with integer support.",
            ),
        )
    end
    length(xs) == 1 && p == 1 && return first(xs) - μ

    b = _uniform_orderstatistic(d)
    cx = cdf(d.dist, first(xs) - 1)
    c = cdf(b, cx)
    mp = zero(first(xs)) * c
    for x in xs
        cx += pdf(d.dist, x)
        c_new = cdf(b, cx)
        mp += (x - μ)^p * (c_new - c)
        c = c_new
    end

    return mp
end

_issymmetric(d) = false
_issymmetric(::Normal) = true
_issymmetric(::NormalCanon) = true
_issymmetric(::Laplace) = true
_issymmetric(::Arcsine) = true
_issymmetric(::TDist) = true
_issymmetric(d::Beta) = d.α == d.β
_issymmetric(::Biweight) = true
_issymmetric(::Triweight) = true
_issymmetric(::SymTriangularDist) = true

## Uniform

mean(d::OrderStatistic{<:Uniform}) = d.rank * scale(d.dist) / (d.n + 1) + minimum(d)
std(d::OrderStatistic{<:Uniform}) = std(_uniform_orderstatistic(d)) * scale(d.dist)
var(d::OrderStatistic{<:Uniform}) = var(_uniform_orderstatistic(d)) * scale(d.dist)^2
skewness(d::OrderStatistic{<:Uniform}) = skewness(_uniform_orderstatistic(d))
kurtosis(d::OrderStatistic{<:Uniform}) = kurtosis(_uniform_orderstatistic(d))

## Exponential

function mean(d::OrderStatistic{<:Exponential})
    # Arnold, eq 4.6.6
    θ = scale(d.dist)
    T = float(typeof(one(θ)))
    return θ * _harmonicdiff(T, d.n - d.rank, d.n)
end
function var(d::OrderStatistic{<:Exponential})
    # Arnold, eq 4.6.7
    θ = scale(d.dist)
    T = float(typeof(one(θ)))
    return θ^2 * _polygamma_diff(T, 1, d.n + 1 - d.rank, d.n + 1)
end
function skewness(d::OrderStatistic{<:Exponential})
    θ = scale(d.dist)
    T = float(typeof(one(θ)))
    return -_polygamma_diff(T, 2, d.n + 1 - d.rank, d.n + 1) /
           _polygamma_diff(T, 1, d.n + 1 - d.rank, d.n + 1)^(3//2)
end
function kurtosis(d::OrderStatistic{<:Exponential})
    θ = scale(d.dist)
    T = float(typeof(one(θ)))
    return _polygamma_diff(T, 3, d.n + 1 - d.rank, d.n + 1) /
           _polygamma_diff(T, 1, d.n + 1 - d.rank, d.n + 1)^2
end

## Logistic

function mean(d::OrderStatistic{<:Logistic})
    # Arnold, eq 4.8.6
    T = typeof(oneunit(partype(d.dist)))
    return scale(d.dist) * _harmonicdiff(T, d.n - d.rank, d.rank - 1) + mean(d.dist)
end
function var(d::OrderStatistic{<:Logistic})
    # Arnold, eq 4.8.7
    σ = scale(d.dist)
    T = float(typeof(one(σ)))
    return σ^2 * _polygamma_sum(T, 1, d.n + 1 - d.rank, d.rank)
end
function skewness(d::OrderStatistic{<:Logistic})
    σ = scale(d.dist)
    T = float(typeof(one(σ)))
    return _polygamma_diff(T, 2, d.rank, d.n + 1 - d.rank) /
           _polygamma_sum(T, 1, d.rank, d.n + 1 - d.rank)^(3//2)
end
function kurtosis(d::OrderStatistic{<:Logistic})
    σ = scale(d.dist)
    T = float(typeof(one(σ)))
    return _polygamma_sum(T, 3, d.rank, d.n + 1 - d.rank) /
           _polygamma_sum(T, 1, d.rank, d.n + 1 - d.rank)^2
end

## Normal

function mean(d::OrderStatistic{<:Normal})
    n = d.n
    n > 5 && return _moment(d, 1)
    rank = d.rank
    μ = mean(d.dist)
    σ = scale(d.dist)
    T = float(typeof(one(σ)))
    # Arnold §4.9
    isodd(n) && rank == (n + 1) ÷ 2 && return μ
    n == 2 && return (2 * rank - 3) * σ / sqrtπ + μ
    n == 3 && return (rank - 2) * 3σ / 2 / sqrtπ + μ
    I2 = atan(T(sqrt2)) / π
    I3 = (6I2 - 1) / 4
    r = max(rank, n - rank + 1)
    s = (-1)^(d.rank != r)
    if n == 4
        c = 6s * σ / sqrtπ
        r == 4 && return c * I2 + μ
        r == 5 && return c * (1 - 3I2) + μ
    end
    if n == 5
        c = 10s * σ / sqrtπ
        r == 5 && return c * I3 + μ
        r == 4 && return c * (3I2 - 4I3) + μ
    end
end

## AffineDistribution

function mean(d::OrderStatistic{<:AffineDistribution})
    σ = scale(d.dist)
    r = σ ≥ 0 ? d.rank : d.n - d.rank + 1
    dρ = OrderStatistic(d.dist.ρ, d.n, r; check_args=false)
    return mean(dρ) * σ + d.dist.μ
end
function var(d::OrderStatistic{<:AffineDistribution})
    σ = scale(d.dist)
    r = σ ≥ 0 ? d.rank : d.n - d.rank + 1
    dρ = OrderStatistic(d.dist.ρ, d.n, r; check_args=false)
    return var(dρ) * σ^2
end
function skewness(d::OrderStatistic{<:AffineDistribution})
    σ = scale(d.dist)
    r = σ ≥ 0 ? d.rank : d.n - d.rank + 1
    return sign(σ) * skewness(OrderStatistic(d.dist.ρ, d.n, r; check_args=false))
end
function kurtosis(d::OrderStatistic{<:AffineDistribution})
    r = scale(d.dist) ≥ 0 ? d.rank : d.n - d.rank + 1
    return kurtosis(OrderStatistic(d.dist.ρ, d.n, r; check_args=false))
end

# Common utilities

_harmonicnum(T::Type{<:Real}, n::Int) = _harmonicnum_from(zero(T), 0, n)

function _harmonicnum_from(Hm::Real, m::Int, n::Int)
    # m ≤ n
    (n - m) < 25 && return sum(Base.Fix1(/, one(Hm)), (m + 1):n; init=Hm)
    return digamma(oftype(Hm, n + 1)) + Base.MathConstants.eulergamma
end

function _harmonicdiff(T::Type{<:Real}, m::Int, n::Int)
    # TODO: improve heuristic
    d = n - m
    m, n = minmax(m, n)
    abs(d) < 50 && return sign(d) * sum(Base.Fix1(/, one(T)), (m + 1):n; init=zero(T))
    Hm = _harmonicnum(T, m)
    Hn = _harmonicnum_from(Hm, m, n)
    return sign(d) * (Hn - Hm)
end

function _polygamma_from(m, ϕk::Real, k::Int, n::Int)
    # backwards recurrence is more stable than forwards
    gap = k - n
    gap > 10 || gap < 0 && return polygamma(m, oftype(ϕk, n))
    num = (-1)^(m + 1) * oftype(ϕk, factorial(m))
    f = Base.Fix1(/, num) ∘ Base.Fix2(^, m + 1)
    return sum(f, (k - 1):-1:n; init=ϕk)
end

function _polygamma_diff(T::Type{<:Real}, m::Int, k::Int, n::Int)
    d = n - k
    k, n = minmax(k, n)
    s = -sign(d)
    if abs(d) ≤ 10
        num = (-1)^m * s * T(factorial(m))
        f = Base.Fix1(/, num) ∘ Base.Fix2(^, m + 1)
        return sum(f, k:(n - 1); init=zero(T))
    end
    ϕn = polygamma(m, T(n))
    ϕk = _polygamma_from(m, ϕn, n, k)
    return s * (ϕn - ϕk)
end

function _polygamma_sum(T::Type{<:Real}, m::Int, k::Int, n::Int)
    k, n = minmax(k, n)
    ϕn = polygamma(m, T(n))
    ϕk = _polygamma_from(m, ϕn, n, k)
    return ϕn + ϕk
end
