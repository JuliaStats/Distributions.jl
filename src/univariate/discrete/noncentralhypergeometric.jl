# Noncentral hypergeometric distribution
# TODO: this distribution needs clean-up and testing

abstract type NoncentralHypergeometric{T<:Real} <: DiscreteUnivariateDistribution end

### handling support

@distr_support NoncentralHypergeometric max(d.n - d.nf, 0) min(d.ns, d.n)

# Functions

function quantile(d::NoncentralHypergeometric{T}, q::Real) where T<:Real
    if !(zero(q) <= q <= one(q))
        T(NaN)
    else
        range = support(d)
        if q > 1/2
            q = 1 - q
            range = reverse(range)
        end

        qsum, i = zero(T), 0
        while qsum < q
            i += 1
            qsum += pdf(d, range[i])
        end
        range[i]
    end
end

params(d::NoncentralHypergeometric) = (d.ns, d.nf, d.n, d.ω)
@inline partype(d::NoncentralHypergeometric{T}) where {T<:Real} = T

## Fisher's noncentral hypergeometric distribution

struct FisherNoncentralHypergeometric{T<:Real} <: NoncentralHypergeometric{T}
    ns::Int    # number of successes in population
    nf::Int    # number of failures in population
    n::Int     # sample size
    ω::T # odds ratio

    function FisherNoncentralHypergeometric{T}(ns::Real, nf::Real, n::Real, ω::T) where T
        @check_args(FisherNoncentralHypergeometric, ns >= zero(ns) && nf >= zero(nf))
        @check_args(FisherNoncentralHypergeometric, zero(n) < n < ns + nf)
        @check_args(FisherNoncentralHypergeometric, ω > zero(ω))
        new{T}(ns, nf, n, ω)
    end
end

FisherNoncentralHypergeometric(ns::Integer, nf::Integer, n::Integer, ω::T) where {T<:Real} = FisherNoncentralHypergeometric{T}(ns, nf, n, ω)

FisherNoncentralHypergeometric(ns::Integer, nf::Integer, n::Integer, ω::Integer) = FisherNoncentralHypergeometric(ns, nf, n, Float64(ω))

# Conversions
convert(::Type{FisherNoncentralHypergeometric{T}}, ns::Real, nf::Real, n::Real, ω::Real) where {T<:Real} = FisherNoncentralHypergeometric(ns, nf, n, T(ω))
convert(::Type{FisherNoncentralHypergeometric{T}}, d::FisherNoncentralHypergeometric{S}) where {T<:Real, S<:Real} = FisherNoncentralHypergeometric(d.ns, d.nf, d.n, T(d.ω))

function _mode(d::FisherNoncentralHypergeometric)
    A = d.ω - 1
    B = d.n - d.nf - (d.ns + d.n + 2)*d.ω
    C = (d.ns + 1)*(d.n + 1)*d.ω
    return -2C / (B - sqrt(B^2 - 4A*C))
end

mode(d::FisherNoncentralHypergeometric) = floor(Int, _mode(d))

testfd(d::FisherNoncentralHypergeometric) = d.ω^3

# The pdf, cdf, mean, and var functions are based on
#
# Liao, J. G; Rosen, Ori (2001). Fast and Stable Algorithms for Computing and Sampling From the
# Noncentral Hypergeometric Distribution. The American Statistician, 55(4), 366–369.
# doi:10.1198/000313001753272547
#
# but the rule for terminating the summation has been slightly modified.
function pdf(d::FisherNoncentralHypergeometric, k::Integer)
    ω, _ = promote(d.ω, float(k))
    l = max(0, d.n - d.nf)
    u = min(d.ns, d.n)
    if !insupport(d, k)
        return zero(ω)
    end
    η = mode(d)
    s = one(ω)
    fᵢ = one(ω)
    fₖ = one(ω)
    for i in (η + 1):u
        rᵢ = (d.ns - i + 1)*ω/(i*(d.nf - d.n  + i))*(d.n - i + 1)
        fᵢ *= rᵢ

        # break if terms no longer contribute to s
        sfᵢ = s + fᵢ
        if sfᵢ == s && i > k
            break
        end
        s = sfᵢ

        if i == k
            fₖ = fᵢ
        end
    end
    fᵢ = one(ω)
    for i in (η - 1):-1:l
        rᵢ₊ = (d.ns - i)*ω/((i + 1)*(d.nf - d.n  + i + 1))*(d.n - i)
        fᵢ /= rᵢ₊

        # break if terms no longer contribute to s
        sfᵢ = s + fᵢ
        if sfᵢ == s && i < k
            break
        end
        s = sfᵢ

        if i == k
            fₖ = fᵢ
        end
    end

    return fₖ/s
end

logpdf(d::FisherNoncentralHypergeometric, k::Integer) = log(pdf(d, k))

function cdf(d::FisherNoncentralHypergeometric, k::Integer)
    ω, _ = promote(d.ω, float(k))
    l = max(0, d.n - d.nf)
    u = min(d.ns, d.n)
    if k < l
        return zero(ω)
    elseif k >= u
        return one(ω)
    end
    η = mode(d)
    s = one(ω)
    fᵢ = one(ω)
    Fₖ = k >= η ? one(ω) : zero(ω)
    for i in (η + 1):u
        rᵢ = (d.ns - i + 1)*ω/(i*(d.nf - d.n  + i))*(d.n - i + 1)
        fᵢ *= rᵢ

        # break if terms no longer contribute to s
        sfᵢ = s + fᵢ
        if sfᵢ == s && i > k
            break
        end
        s = sfᵢ
        if i <= k
            Fₖ += fᵢ
        end
    end
    fᵢ = one(ω)
    for i in (η - 1):-1:l
        rᵢ₊ = (d.ns - i)*ω/((i + 1)*(d.nf - d.n + i + 1))*(d.n - i)
        fᵢ /= rᵢ₊

        # break if terms no longer contribute to s
        sfᵢ = s + fᵢ
        if sfᵢ == s && i < k
            break
        end
        s = sfᵢ
        if i <= k
            Fₖ += fᵢ
        end
    end

    return Fₖ/s
end

logcdf(d::FisherNoncentralHypergeometric, k::Integer) = log(cdf(d, k))

function _expectation(f, d::FisherNoncentralHypergeometric)
    ω = float(d.ω)
    l = max(0, d.n - d.nf)
    u = min(d.ns, d.n)
    η = mode(d)
    s = one(ω)
    m = f(η)*s
    fᵢ = one(ω)
    for i in (η + 1):u
        rᵢ = (d.ns - i + 1)*ω/(i*(d.nf - d.n  + i))*(d.n - i + 1)
        fᵢ *= rᵢ

        # break if terms no longer contribute to s
        sfᵢ = s + fᵢ
        if sfᵢ == s
            break
        end
        s = sfᵢ

        m += f(i)*fᵢ
    end
    fᵢ = one(ω)
    for i in (η - 1):-1:l
        rᵢ₊ = (d.ns - i)*ω/((i + 1)*(d.nf - d.n  + i + 1))*(d.n - i)
        fᵢ /= rᵢ₊

        # break if terms no longer contribute to s
        sfᵢ = s + fᵢ
        if sfᵢ == s
            break
        end
        s = sfᵢ

        m += f(i)*fᵢ
    end

    return m/s
end

mean(d::FisherNoncentralHypergeometric) = _expectation(identity, d)

function var(d::FisherNoncentralHypergeometric)
    μ = mean(d)
    return _expectation(t -> (t - μ)^2, d)
end

## Wallenius' noncentral hypergeometric distribution

struct WalleniusNoncentralHypergeometric{T<:Real} <: NoncentralHypergeometric{T}
    ns::Int    # number of successes in population
    nf::Int    # number of failures in population
    n::Int     # sample size
    ω::T # odds ratio

    function WalleniusNoncentralHypergeometric{T}(ns::Real, nf::Real, n::Real, ω::T) where T
        @check_args(WalleniusNoncentralHypergeometric, ns >= zero(ns) && nf >= zero(nf))
        @check_args(WalleniusNoncentralHypergeometric, zero(n) < n < ns + nf)
        @check_args(WalleniusNoncentralHypergeometric, ω > zero(ω))
        new{T}(ns, nf, n, ω)
    end
end

WalleniusNoncentralHypergeometric(ns::Integer, nf::Integer, n::Integer, ω::T) where {T<:Real} = WalleniusNoncentralHypergeometric{T}(ns, nf, n, ω)

WalleniusNoncentralHypergeometric(ns::Integer, nf::Integer, n::Integer, ω::Integer) = WalleniusNoncentralHypergeometric(ns, nf, n, Float64(ω))

# Conversions
convert(::Type{WalleniusNoncentralHypergeometric{T}}, ns::Real, nf::Real, n::Real, ω::Real) where {T<:Real} = WalleniusNoncentralHypergeometric(ns, nf, n, T(ω))
convert(::Type{WalleniusNoncentralHypergeometric{T}}, d::WalleniusNoncentralHypergeometric{S}) where {T<:Real, S<:Real} = WalleniusNoncentralHypergeometric(d.ns, d.nf, d.n, T(d.ω))

# Properties
mean(d::WalleniusNoncentralHypergeometric) = sum(support(d) .* pdf.(Ref(d), support(d)))
var(d::WalleniusNoncentralHypergeometric)  = sum((support(d) .- mean(d)).^2 .* pdf.(Ref(d), support(d)))
mode(d::WalleniusNoncentralHypergeometric) = support(d)[argmax(pdf.(Ref(d), support(d)))]

entropy(d::WalleniusNoncentralHypergeometric) = 1

testfd(d::WalleniusNoncentralHypergeometric) = d.ω^3

function logpdf(d::WalleniusNoncentralHypergeometric, k::Real)
    if insupport(d, k)
        D = d.ω * (d.ns - k) + (d.nf - d.n + k)
        f(t) = (1 - t^(d.ω / D))^k * (1 - t^(1 / D))^(d.n - k)
        I, _ = quadgk(f, 0, 1)
        return -log(d.ns + 1) - logbeta(d.ns - k + 1, k + 1) -
            log(d.nf + 1) - logbeta(d.nf - d.n + k + 1, d.n - k + 1) + log(I)
    else
        return log(zero(k))
    end
end
