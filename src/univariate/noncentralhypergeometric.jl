# Noncentral hypergeometric distribution

abstract NoncentralHypergeometric <: DiscreteUnivariateDistribution

### handling support

function insupport(d::NoncentralHypergeometric, x::Real)
    isinteger(x) && minimum(d) <= x <= maximum(d)
end

isupperbounded(::Union(NoncentralHypergeometric, Type{NoncentralHypergeometric})) = true
islowerbounded(::Union(NoncentralHypergeometric, Type{NoncentralHypergeometric})) = true
isbounded(::Union(NoncentralHypergeometric, Type{NoncentralHypergeometric})) = true

minimum(d::NoncentralHypergeometric) = max(0, d.n-d.nf)
maximum(d::NoncentralHypergeometric) = min(d.n, d.ns)
support(d::NoncentralHypergeometric) = minimum(d):maximum(d)

# Functions

pdf(d::NoncentralHypergeometric, k::Real) = exp(logpdf(d,k))
logpdf(d::NoncentralHypergeometric, k::Real) = log(pdf(d,k))
cdf(d::NoncentralHypergeometric, k::Real) = sum([pdf(d,i) for i in minimum(d):k])

function quantile(d::NoncentralHypergeometric, q::Real)
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

## Fisher's noncentral hypergeometric distribution

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
mode(d::FisherNoncentralHypergeometric) = int(floor(_mode(d)))

function logpdf(d::FisherNoncentralHypergeometric, k::Real)
    isinteger(k) || return 0.0
    log(binomial(d.ns, k)) + log(binomial(d.nf, d.n-k)) + k*log(d.ω) - log(_P(d,0))
end

## Wallenius' noncentral hypergeometric distribution

immutable WalleniusNoncentralHypergeometric <: NoncentralHypergeometric
    ns::Int    # number of successes in population
    nf::Int    # number of failures in population
    n::Int     # sample size
    ω::Float64 # odds ratio
    function WalleniusNoncentralHypergeometric(s::Real, f::Real, n::Real, ω::Float64)
        isinteger(s) && zero(s) <= s || error("ns must be a non-negative integer")
        isinteger(f) && zero(f) <= f || error("nf must be a non-negative integer")
        isinteger(n) && zero(n) < n < s + f ||
            error("n must be a positive integer <= (ns + nf)")
        zero(ω) < ω || error("ω must be a positive real value")
        new(float64(s), float64(f), float64(n), float64(ω))
    end
end

# Properties
mean(d::WalleniusNoncentralHypergeometric)=sum(support(d).*pdf(d,support(d)))
var(d::WalleniusNoncentralHypergeometric) = sum((support(d)-mean(d)).^2.*pdf(d,support(d)))
mode(d::WalleniusNoncentralHypergeometric) = support(d)[indmax(pdf(d,support(d)))]

function pdf(d::WalleniusNoncentralHypergeometric, k::Real)
    D = d.ω*(d.ns-k)+(d.nf-d.n+k)
    f(t) = (1-t^(d.ω/D))^k * (1-t^(1/D))^(d.n-k)
    I,_ = quadgk(f,0,1)
    binomial(d.ns,k)*binomial(d.nf,d.n-k)*I
end