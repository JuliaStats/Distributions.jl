# Noncentral hypergeometric distribution
# TODO: this distribution needs clean-up and testing

abstract NoncentralHypergeometric <: DiscreteUnivariateDistribution

### handling support

function insupport(d::NoncentralHypergeometric, x::Real)
    isinteger(x) && minimum(d) <= x <= maximum(d)
end

@distr_support NoncentralHypergeometric max(d.n - d.nf, 0) min(d.ns, d.n)

# Functions

function quantile(d::NoncentralHypergeometric, q::Float64)
    if !(zero(q) <= q <= one(q))
        NaN
    else
        range = support(d)
        if q > 0.5
            q = 1-q
            range = reverse(range)
        end

        qsum, i = 0.0, 0
        while qsum < q
            i += 1
            qsum += pdf(d, range[i])
        end
        range[i]
    end
end

params(d::NoncentralHypergeometric) = (d.ns, d.nf, d.n, d.ω)

## Fisher's noncentral hypergeometric distribution

immutable FisherNoncentralHypergeometric <: NoncentralHypergeometric
    ns::Int    # number of successes in population
    nf::Int    # number of failures in population
    n::Int     # sample size
    ω::Float64 # odds ratio

    function FisherNoncentralHypergeometric(ns::Real, nf::Real, n::Real, ω::Float64)
        @check_args(FisherNoncentralHypergeometric, ns >= zero(ns) && nf >= zero(nf))
        @check_args(FisherNoncentralHypergeometric, zero(n) < n < ns + nf)
        @check_args(FisherNoncentralHypergeometric, ω > zero(ω))
        new(ns, nf, n, ω)
    end
end

# Properties
function _P(d::FisherNoncentralHypergeometric, k::Int)
    y = support(d)
    p = -log(d.ns + 1) - lbeta(d.ns - y + 1, y + 1) -
            log(d.nf + 1) - lbeta(d.nf - d.n + y + 1, d.n - y + 1) +
            xlogy(y, d.ω) + xlogy(k, y)
    logsumexp(p)
end

function _mode(d::FisherNoncentralHypergeometric)
    A = d.ω - 1
    B = d.n - d.nf - (d.ns + d.n + 2)*d.ω
    C = (d.ns + 1)*(d.n + 1)*d.ω
    -2*C / (B - sqrt(B^2-4*A*C))
end

mean(d::FisherNoncentralHypergeometric) = exp(_P(d,1) - _P(d,0))
var(d::FisherNoncentralHypergeometric) = exp(_P(d,2) - _P(d,0)) - exp(2*(_P(d,1) - _P(d,0)))
mode(d::FisherNoncentralHypergeometric) = floor(Int, _mode(d))

logpdf(d::FisherNoncentralHypergeometric, k::Int) =
    -log(d.ns + 1) - lbeta(d.ns - k + 1, k + 1) -
    log(d.nf + 1) - lbeta(d.nf - d.n + k + 1, d.n - k + 1) +
    xlogy(k, d.ω) - _P(d, 0)

pdf(d::FisherNoncentralHypergeometric, k::Int) = exp(logpdf(d, k))


## Wallenius' noncentral hypergeometric distribution

immutable WalleniusNoncentralHypergeometric <: NoncentralHypergeometric
    ns::Int    # number of successes in population
    nf::Int    # number of failures in population
    n::Int     # sample size
    ω::Float64 # odds ratio
    function WalleniusNoncentralHypergeometric(ns::Real, nf::Real, n::Real, ω::Float64)
        @check_args(WalleniusNoncentralHypergeometric, ns >= zero(ns) && nf >= zero(nf))
        @check_args(WalleniusNoncentralHypergeometric, zero(n) < n < ns + nf)
        @check_args(WalleniusNoncentralHypergeometric, ω > zero(ω))
        new(ns, nf, n, ω)
    end
end

# Properties
mean(d::WalleniusNoncentralHypergeometric) = sum(support(d) .* pdf(d, support(d)))
var(d::WalleniusNoncentralHypergeometric)  = sum((support(d) - mean(d)).^2 .* pdf(d, support(d)))
mode(d::WalleniusNoncentralHypergeometric) = support(d)[indmax(pdf(d, support(d)))]

function logpdf(d::WalleniusNoncentralHypergeometric, k::Int)
    D = d.ω * (d.ns - k) + (d.nf - d.n + k)
    f(t) = (1 - t^(d.ω / D))^k * (1 - t^(1 / D))^(d.n - k)
    I, _ = quadgk(f, 0, 1)
    return -log(d.ns + 1) - lbeta(d.ns - k + 1, k + 1) -
    log(d.nf + 1) - lbeta(d.nf - d.n + k + 1, d.n - k + 1) + log(I)
end

pdf(d::WalleniusNoncentralHypergeometric, k::Int) = exp(logpdf(d, k))
