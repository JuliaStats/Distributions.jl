immutable Hypergeometric <: DiscreteUnivariateDistribution
    ns::Int # number of successes in population
    nf::Int # number of failures in population
    n::Int  # sample size
    function Hypergeometric(s::Real, f::Real, n::Real)
        isinteger(s) && zero(s) <= s || error("ns must be a non-negative integer")
        isinteger(f) && zero(f) <= f || error("nf must be a non-negative integer")
        isinteger(n) && zero(n) < n < s + f ||
            error("n must be a positive integer <= (ns + nf)")
        new(int(s), int(f), int(n))
    end
end

isupperbounded(d::Union(Hypergeometric, Type{Hypergeometric})) = true
islowerbounded(d::Union(Hypergeometric, Type{Hypergeometric})) = true
isbounded(d::Union(Hypergeometric, Type{Hypergeometric})) = true

minimum(d::Hypergeometric) = max(0,d.n-d.nf)
maximum(d::Hypergeometric) = min(d.n,d.ns)
support(d::Hypergeometric) = minimum(d):maximum(d)

function insupport(d::Hypergeometric, x::Number)
    isinteger(x) && zero(x) <= x <= d.n && (d.n - d.nf) <= x <= d.ns
end

mean(d::Hypergeometric) = d.n * d.ns / (d.ns + d.nf)

mode(d::Hypergeometric) = floor((d.n+1)*(d.ns+1)/(d.ns+d.nf+2))

function var(d::Hypergeometric)
    N = d.ns + d.nf
    d.n * (d.ns / N) * (d.nf / N) * ((N - d.n) / (N - 1.0))
end

function skewness(d::Hypergeometric)
    N = d.ns + d.nf
    (d.nf-d.ns)*((N-2d.n)/(N-2))*sqrt((N-1)/(d.n*d.ns*d.nf*(N-d.n)))
end

function kurtosis(d::Hypergeometric)
    N = d.ns + d.nf
    ((N-1)*N^2*(N*(N+1)-6*d.ns*d.nf-6*d.n*(N-d.n))+6d.n*d.ns*d.nf*(N-d.n)*(5N-6))/
    (d.n*d.ns*d.nf*(N-d.n)*(N-2)*(N-3))
end

function entropy(d::Hypergeometric)
    e = 0.0
    for x = support(d)
        p = pdf(d,x)
        e -= log(p)*p
    end
    e
end


function pdf(d::Hypergeometric, x::Real) 
    N = d.ns + d.nf
    p = d.ns / N
    pdf(Binomial(d.ns,p),x) * pdf(Binomial(d.nf,p),d.n-x) / pdf(Binomial(N,p),d.n)
end

function cdf(d::Hypergeometric, x::Real)
    if x < minimum(d)
        return 0.0
    elseif x >= maximum(d)
        return 1.0
    end
    p = 0.0
    for i = minimum(d):floor(x)
        p += pdf(d,i)
    end
    p
end

function quantile(d::Hypergeometric, p::Real)
    if p < 0 || p > 1 return NaN end
    if p == 0 return minimum(d) end
    if p == 1 return maximum(d) end
    cp = 0.0
    for x = support(d)
        cp += pdf(d,x)
        if cp >= p
            return x
        end
    end
end

# TODO: Implement:
#   V. Kachitvichyanukul & B. Schmeiser
#   "Computer generation of hypergeometric random variates"
#   Journal of Statistical Computation and Simulation, 22(2):127-145
#   doi:10.1080/00949658508810839
rand(d::Hypergeometric) = quantile(d,rand())
