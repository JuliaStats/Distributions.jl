# TODO: this distribution may need clean-up
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
struct PoissonBinomial{T<:Real} <: DiscreteUnivariateDistribution
    p::Vector{T}
    pmf::Vector{T}

    function PoissonBinomial{T}(p::AbstractArray) where {T <: Real}
        pb = poissonbinomial_pdf_fft(p)
        @assert isprobvec(pb)
        new{T}(p, pb)
    end
end

function PoissonBinomial(p::AbstractArray{T}; check_args=true) where {T <: Real}
    if check_args
        for i in eachindex(p)
            @check_args(PoissonBinomial, 0 <= p[i] <= 1)
        end
    end
    return PoissonBinomial{T}(p)
end

@distr_support PoissonBinomial 0 length(d.p)

#### Conversions

function PoissonBinomial(::Type{PoissonBinomial{T}}, p::Vector{S}) where {T, S}
    return PoissonBinomial(Vector{T}(p))
end
function PoissonBinomial(::Type{PoissonBinomial{T}}, d::PoissonBinomial{S}) where {T, S}
    return PoissonBinomial(Vector{T}(d.p), check_args=false)
end

#### Parameters

ntrials(d::PoissonBinomial) = length(d.p)
succprob(d::PoissonBinomial) = d.p
failprob(d::PoissonBinomial{T}) where {T} = one(T) .- d.p

params(d::PoissonBinomial) = (d.p,)
partype(::PoissonBinomial{T}) where {T} = T

#### Properties

mean(d::PoissonBinomial) = sum(succprob(d))
var(d::PoissonBinomial) = sum(succprob(d) .* failprob(d))

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

entropy(d::PoissonBinomial) = entropy(Categorical(d.pmf))
median(d::PoissonBinomial) = median(Categorical(d.pmf)) - 1
mode(d::PoissonBinomial) = argmax(d.pmf) - 1
modes(d::PoissonBinomial) = [x  - 1 for x in modes(Categorical(d.pmf))]

#### Evaluation

quantile(d::PoissonBinomial, x::Float64) = quantile(Categorical(d.pmf), x) - 1

function mgf(d::PoissonBinomial{T}, t::Real) where {T}
    p,  = params(d)
    prod(one(T) .- p .+ p .* exp(t))
end

function cf(d::PoissonBinomial{T}, t::Real) where {T}
    p,  = params(d)
    prod(one(T) .- p .+ p .* cis(t))
end

pdf(d::PoissonBinomial, k::Integer) = insupport(d, k) ? d.pmf[k+1] : 0
function logpdf(d::PoissonBinomial{T}, k::Int) where T<:Real
    insupport(d, k) ? log(d.pmf[k + 1]) : -T(Inf)
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
