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
binomial{T<:Integer}(n::T, k::T) = Base.binomial(BigInt(n), BigInt(k))
binomial{T<:Integer}(n::T, kR::OrdinalRange{T}) = [binomial(n,k) for k in kR]
function _P(d::FisherNoncentralHypergeometric, k::Int)
    y = support(d)
    p = binomial(d.ns, y) .* binomial(d.nf, d.n-y) .* d.ω.^y .* y.^k
    sum(p)
end

function _mode(d::FisherNoncentralHypergeometric)
    A = d.ω - 1
    B = d.n - d.nf - (d.ns + d.n + 2)*d.ω
    C = (d.ns + 1)*(d.n + 1)*d.ω
    -2*C / (B - sqrt(B^2-4*A*C))
end

mean(d::FisherNoncentralHypergeometric) =_P(d,1) / _P(d,0)
var(d::FisherNoncentralHypergeometric) = _P(d,2)/_P(d,0) - (_P(d,1) / _P(d,0))^2
@compat mode(d::FisherNoncentralHypergeometric) = floor(Int, _mode(d))

@compat logpdf(d::FisherNoncentralHypergeometric, k::Int) =
    Float64(log(binomial(d.ns, k)) + log(binomial(d.nf, d.n-k)) + k*log(d.ω) - log(_P(d,0)))

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
mean(d::WalleniusNoncentralHypergeometric)=sum(support(d).*pdf(d,support(d)))
var(d::WalleniusNoncentralHypergeometric) = sum((support(d)-mean(d)).^2.*pdf(d,support(d)))
mode(d::WalleniusNoncentralHypergeometric) = support(d)[indmax(pdf(d,support(d)))]

function pdf(d::WalleniusNoncentralHypergeometric, k::Int)
    D = d.ω*(d.ns-k)+(d.nf-d.n+k)
    f(t) = (1-t^(d.ω/D))^k * (1-t^(1/D))^(d.n-k)
    I,_ = quadgk(f,0,1)
    @compat Float64(binomial(d.ns,k)*binomial(d.nf,d.n-k)*I)
end

logpdf(d::WalleniusNoncentralHypergeometric, k::Int) = log(pdf(d, k))
