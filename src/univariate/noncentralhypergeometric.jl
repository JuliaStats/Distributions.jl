# Noncentral hypergeometric distribution

abstract NoncentralHypergeometric <: DiscreteUnivariateDistribution

immutable FisherNoncentralHypergeometric <: NoncentralHypergeometric
    ns::Int    # number of successes in population
    nf::Int    # number of failures in population
    n::Int     # sample size
    ω::Float64 # odds ratio
    function FisherNoncentralHypergeometric(s::Real, f::Real, n::Real, ω::Float64)
        isinteger(s) && zero(s) <= s || error("ns must be a non-negative integer")
        isinteger(f) && zero(f) <= f || error("nf must be a non-negative integer")
        isinteger(n) && zero(n) < n < s + f ||
            error("n must be a positive integer <= (ns + nf)")
        zero(ω) < ω || error("ω must be a positive real value")
        new(float64(s), float64(f), float64(n), float64(ω))
    end
end

### handling support

function insupport(d::FisherNoncentralHypergeometric, x::Real)
    isinteger(x) && minimum(d) <= x <= maximum(d)
end

isupperbounded(::Union(FisherNoncentralHypergeometric, Type{FisherNoncentralHypergeometric})) = true
islowerbounded(::Union(FisherNoncentralHypergeometric, Type{FisherNoncentralHypergeometric})) = true
isbounded(::Union(FisherNoncentralHypergeometric, Type{FisherNoncentralHypergeometric})) = true

minimum(d::FisherNoncentralHypergeometric) = max(0, d.n-d.nf)
maximum(d::FisherNoncentralHypergeometric) = min(d.n, d.ns)
support(d::FisherNoncentralHypergeometric) = minimum(d):maximum(d)

## Properties
binomial{T<:Integer}(n::T, k::T) = Base.binomial(BigInt(n), BigInt(k))
binomial{T<:Integer}(n::T, kR::OrdinalRange{T}) = [binomial(n,k) for k in kR]
function _P(d::FisherNoncentralHypergeometric, k::Int)
    y = support(d)
    p = binomial(d.ns, y) .* binomial(d.nf, d.n-y) .* d.ω.^y .* y.^k
    sum(p)
end

mean(d::FisherNoncentralHypergeometric) = _P(d,1) / _P(d,0)
var(d::FisherNoncentralHypergeometric) = _P(d,2)/_P(d,0) - (_P(d,1) / _P(d,0))^2

function mode(d::FisherNoncentralHypergeometric)
    A = d.ω - 1
    B = d.n - d.nf - (d.ns + d.n + 2)*d.ω
    C = (d.ns + 1)*(d.n + 1)*d.ω
    int(floor(-2*C / (B - sqrt(B^2-4*A*C))))
end

## Functions
function logpdf(d::FisherNoncentralHypergeometric, k::Real)
    isinteger(k) || return 0.0
    log(binomial(d.ns, k)) + log(binomial(d.nf, d.n-k)) + k*log(d.ω) - log(_P(d,0))
end

pdf(d::FisherNoncentralHypergeometric, k::Real) = exp(logpdf(d,k))
cdf(d::FisherNoncentralHypergeometric, k::Real) = sum([pdf(d,i) for i in minimum(d):k])

function quantile(d::FisherNoncentralHypergeometric, q::Real)
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