"""
    PoissonBinomial(p)

A *Poisson-binomial distribution* describes the number of successes in a sequence of independent trials, wherein each trial has a different success rate.
It is parameterized by a vector `p` (of length ``K``), where ``K`` is the total number of trials and `p[i]` corresponds to the probability of success of the `i`th trial.

```math
P(X = k) = \\sum\\limits_{A\\in F_k} \\prod\\limits_{i\\in A} p[i] \\prod\\limits_{j\\in A^c} (1-p[j]), \\quad \\text{ for } k = 0,1,2,\\ldots,K,
```

where ``F_k`` is the set of all subsets of ``k`` integers that can be selected from ``\\{1,2,3,...,K\\}``.

```julia
PoissonBinomial(p)   # Poisson Binomial distribution with success rate vector p

params(d)            # Get the parameters, i.e. (p,)
succprob(d)          # Get the vector of success rates, i.e. p
failprob(d)          # Get the vector of failure rates, i.e. 1-p
```

External links:

* [Poisson-binomial distribution on Wikipedia](http://en.wikipedia.org/wiki/Poisson_binomial_distribution)

"""
mutable struct PoissonBinomial{T<:Real,P<:AbstractVector{T}} <: DiscreteUnivariateDistribution
    p::P
    pmf::Union{Nothing,Vector{T}} # lazy computation of the probability mass function

    function PoissonBinomial{T}(p::AbstractVector{T}; check_args::Bool=true) where {T <: Real}
        @check_args(
            PoissonBinomial,
            (
                p,
                all(x -> zero(x) <= x <= one(x), p),
                "p must be a vector of success probabilities",
            ),
        )
        return new{T,typeof(p)}(p, nothing)
    end
end

function PoissonBinomial(p::AbstractVector{T}; check_args::Bool=true) where {T<:Real}
    return PoissonBinomial{T}(p; check_args=check_args)
end

function Base.getproperty(d::PoissonBinomial, x::Symbol)
    if x === :pmf
        z = getfield(d, :pmf)
        if z === nothing
            y = poissonbinomial_pdf(d.p)
            isprobvec(y) || error("probability mass function is not normalized")
            setfield!(d, :pmf, y)
            return y
        else
            return z
        end
    else
        return getfield(d, x)
    end
end

@distr_support PoissonBinomial 0 length(d.p)

#### Conversions

function PoissonBinomial(::Type{PoissonBinomial{T}}, p::AbstractVector{S}) where {T, S}
    return PoissonBinomial(AbstractVector{T}(p))
end
function PoissonBinomial(::Type{PoissonBinomial{T}}, d::PoissonBinomial{S}) where {T, S}
    return PoissonBinomial(AbstractVector{T}(d.p), check_args=false)
end

#### Parameters

ntrials(d::PoissonBinomial) = length(d.p)
succprob(d::PoissonBinomial) = d.p
failprob(d::PoissonBinomial{T}) where {T} = one(T) .- d.p

params(d::PoissonBinomial) = (d.p,)
partype(::PoissonBinomial{T}) where {T} = T

#### Properties

mean(d::PoissonBinomial) = sum(succprob(d))
var(d::PoissonBinomial) = sum(p * (1 - p) for p in succprob(d))

function skewness(d::PoissonBinomial{T}) where {T}
    v = zero(T)
    s = zero(T)
    p,  = params(d)
    for i in eachindex(p)
        v += p[i] * (one(T) - p[i])
        s += p[i] * (one(T) - p[i]) * (one(T) - T(2) * p[i])
    end
    return s / sqrt(v) / v
end

function kurtosis(d::PoissonBinomial{T}) where {T}
    v = zero(T)
    s = zero(T)
    p,  = params(d)
    for i in eachindex(p)
        v += p[i] * (one(T) - p[i])
        s += p[i] * (one(T) - p[i]) * (one(T) - T(6) * (one(T) - p[i]) * p[i])
    end
    s / v / v
end

entropy(d::PoissonBinomial) = entropy(d.pmf)
median(d::PoissonBinomial) = median(Categorical(d.pmf)) - 1
mode(d::PoissonBinomial) = argmax(d.pmf) - 1
modes(d::PoissonBinomial) = modes(DiscreteNonParametric(support(d), d.pmf))

#### Evaluation

quantile(d::PoissonBinomial, x::Float64) = quantile(Categorical(d.pmf), x) - 1

function mgf(d::PoissonBinomial, t::Real)
    expm1_t = expm1(t)
    mapreduce(*, succprob(d)) do p
        1 + p * expm1_t
    end
end

function cf(d::PoissonBinomial, t::Real)
    cis_t = cis(t)
    mapreduce(*, succprob(d)) do p
        1 - p + p * cis_t
    end
end

pdf(d::PoissonBinomial, k::Real) = insupport(d, k) ? d.pmf[Int(k+1)] : zero(eltype(d.pmf))
logpdf(d::PoissonBinomial, k::Real) = log(pdf(d, k))

cdf(d::PoissonBinomial, k::Int) = integerunitrange_cdf(d, k)

# leads to numerically more accurate results
for f in (:ccdf, :logcdf, :logccdf)
    @eval begin
        $f(d::PoissonBinomial, k::Real) = $(Symbol(f, :_int))(d, k)
        $f(d::PoissonBinomial, k::Int) = $(Symbol(:integerunitrange_, f))(d, k)
    end
end

# Computes the pdf of a poisson-binomial random variable using
# simple, fast recursive formula
#
#      Marlin A. Thomas & Audrey E. Taub (1982)
#      Calculating binomial probabilities when the trial probabilities are unequal,
#      Journal of Statistical Computation and Simulation, 14:2, 125-131, DOI: 10.1080/00949658208810534
#
function poissonbinomial_pdf(p)
    S = zeros(eltype(p), length(p) + 1)
    S[1] = 1
    @inbounds for (col, p_col) in enumerate(p)
        q_col = 1 - p_col
        for row in col:(-1):1
            S[row + 1] = q_col * S[row + 1] + p_col * S[row]
        end
        S[1] *= q_col
    end
    return S
end

# Computes the pdf of a poisson-binomial random variable using
# fast fourier transform
#
#     Hong, Y. (2013).
#     On computing the distribution function for the Poisson binomial
#     distribution. Computational Statistics and Data Analysis, 59, 41–51.
#
function poissonbinomial_pdf_fft(p::AbstractArray{T}) where {T <: Real}
    n = length(p)
    ω = 2 * one(T) / (n + 1)

    x = Vector{Complex{T}}(undef, n+1)
    lmax = ceil(Int, n/2)
    x[1] = one(T)/(n + 1)
    for l=1:lmax
        logz = zero(T)
        argz = zero(T)
        for j=1:n
            zjl = 1 - p[j] + p[j] * cospi(ω*l) + im * p[j] * sinpi(ω * l)
            logz += log(abs(zjl))
            argz += atan(imag(zjl), real(zjl))
        end
        dl = exp(logz)
        x[l + 1] = dl * cos(argz) / (n + 1) + dl * sin(argz) * im / (n + 1)
        if n + 1 - l > l
            x[n + 1 - l + 1] = conj(x[l + 1])
        end
    end
    [max(0, real(xi)) for xi in _dft(x)]
end

# A simple implementation of a DFT to avoid introducing a dependency
# on an external FFT package just for this one distribution
function _dft(x::Vector{T}) where T
    n = length(x)
    y = zeros(complex(float(T)), n)
    @inbounds for j = 0:n-1, k = 0:n-1
        y[k+1] += x[j+1] * cis(-π * float(T)(2 * mod(j * k, n)) / n)
    end
    return y
end

#### Sampling

sampler(d::PoissonBinomial) = PoissBinAliasSampler(d)

# Compute matrix of partial derivatives [∂P(X=j-1)/∂pᵢ]_{i=1,…,n; j=1,…,n+1}
#
# This implementation uses the same dynamic programming "trick" as for the computation of
# the primals.
#
# Reference (for the primal):
#
#      Marlin A. Thomas & Audrey E. Taub (1982)
#      Calculating binomial probabilities when the trial probabilities are unequal,
#      Journal of Statistical Computation and Simulation, 14:2, 125-131, DOI: 10.1080/00949658208810534
function poissonbinomial_pdf_partialderivatives(p::AbstractVector{<:Real})
    n = length(p)
    A = zeros(eltype(p), n, n + 1)
    @inbounds for j in 1:n
        A[j, end] = 1
    end
    @inbounds for (i, pi) in enumerate(p)
        qi = 1 - pi
        for k in (n - i + 1):n
            kp1 = k + 1
            for j in 1:(i - 1)
                A[j, k] = pi * A[j, k] + qi * A[j, kp1]
            end
            for j in (i+1):n
                A[j, k] = pi * A[j, k] + qi * A[j, kp1]
            end
        end
        for j in 1:(i-1)
            A[j, end] *= pi
        end
        for j in (i+1):n
            A[j, end] *= pi
        end
    end
    @inbounds for j in 1:n, i in 1:n
        A[i, j] -= A[i, j+1]
    end
    return A
end
