# Noncentral hypergeometric distribution
# TODO: this distribution needs clean-up and testing

abstract type NoncentralHypergeometric{T<:Real} <: DiscreteUnivariateDistribution end

### handling support

function insupport(d::NoncentralHypergeometric, x::Real)
    isinteger(x) && minimum(d) <= x <= maximum(d)
end

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

# Properties
function _P(d::FisherNoncentralHypergeometric, k::Int)
    y = support(d)
    p = -log(d.ns + 1) .- logbeta.(d.ns + 1 .- y, y .+ 1) .-
            log(d.nf + 1) .- logbeta.(d.nf - d.n + 1 .+ y, d.n + 1 .- y) .+
            xlogy.(y, d.ω) .+ xlogy.(k, y)
    logsumexp(p)
end

function _mode(d::FisherNoncentralHypergeometric)
    A = d.ω - 1
    B = d.n - d.nf - (d.ns + d.n + 2)*d.ω
    C = (d.ns + 1)*(d.n + 1)*d.ω
    -2C / (B - sqrt(B^2-4A*C))
end

mean(d::FisherNoncentralHypergeometric) = exp.(_P(d,1) - _P(d,0))
var(d::FisherNoncentralHypergeometric) = exp.(_P(d,2) .- _P(d,0)) .- exp.(2 .* (_P(d,1) .- _P(d,0)))
mode(d::FisherNoncentralHypergeometric) = floor(Int, _mode(d))

testfd(d::FisherNoncentralHypergeometric) = d.ω^3

logpdf(d::FisherNoncentralHypergeometric, k::Int) =
    -log(d.ns + 1) - logbeta(d.ns - k + 1, k + 1) -
    log(d.nf + 1) - logbeta(d.nf - d.n + k + 1, d.n - k + 1) +
    xlogy(k, d.ω) - _P(d, 0)

pdf(d::FisherNoncentralHypergeometric, k::Int) = exp(logpdf(d, k))


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

function logpdf(d::WalleniusNoncentralHypergeometric, k::Int)
    D = d.ω * (d.ns - k) + (d.nf - d.n + k)
    f(t) = (1 - t^(d.ω / D))^k * (1 - t^(1 / D))^(d.n - k)
    I, _ = quadgk(f, 0, 1)
    return -log(d.ns + 1) - logbeta(d.ns - k + 1, k + 1) -
    log(d.nf + 1) - logbeta(d.nf - d.n + k + 1, d.n - k + 1) + log(I)
end

pdf(d::WalleniusNoncentralHypergeometric, k::Int) = exp(logpdf(d, k))
