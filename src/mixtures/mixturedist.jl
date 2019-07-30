# General mixture distributions
"""

  All subtypes of `AbstractMixtureDistribution` must implement the following method:

  - components(d):  return the distribution components

  - probs(d):       return a vector of prior probabilities over components.

  And may implement these:

  - ncomponents(d): the number of components

  - component(d, k):  return the k-th component

Mixture distributions can contain any combination of distributions with countable and uncountable support. All of the distributions must have the same VariateForm, however.
"""
abstract type AbstractMixtureDistribution{VF<:VariateForm,VS<:ValueSupport} <: Distribution{VF, VS} end

"""
    components(d::AbstractMixtureDistribution)

Get a list of components of the mixture distribution `d`.
"""
components(d::AbstractMixtureDistribution)

"""
    components(d::AbstractMixtureDistribution)

Get the number of components of the mixture distribution `d`.
"""
ncomponents(d::AbstractMixtureDistribution) = length(components(d))

"""
    component(d::AbstractMixtureDistribution, k::Integer)

Get the `k'th component of the mixture distribution `d`.
"""
component(d::AbstractMixtureDistribution, k::Integer) = components(d)[k]

"""
    probs(d::AbstractMixtureDistribution)

Get the vector of prior probabilities of all components of `d`.
"""
probs(d::AbstractMixtureDistribution)

const UnivariateMixtureDistribution{S<:ValueSupport} =
    AbstractMixtureDistribution{Univariate,S}
const MultivariateMixtureDistribution{S<:ValueSupport} =
    AbstractMixtureDistribution{Multivariate,S}
const MatrixvariateMixtureDistribution{S<:ValueSupport} =
    AbstractMixtureDistribution{Matrixvariate,S}

minimum(md::AbstractMixtureDistribution) = minimum(minimum.(components(md)))
maximum(md::AbstractMixtureDistribution) = maximum(maximum.(components(md)))

function mean(d::UnivariateMixtureDistribution)
    K = ncomponents(d)
    p = probs(d)
    m = 0.0
    for i = 1:K
        pi = p[i]
        if pi > 0.0
            c = component(d, i)
            m += mean(c) * pi
        end
    end
    return m
end

function mean(d::MultivariateMixtureDistribution)
    K = ncomponents(d)
    p = probs(d)
    m = zeros(length(d))
    for i = 1:K
        pi = p[i]
        if pi > 0.0
            c = component(d, i)
            BLAS.axpy!(pi, mean(c), m)
        end
    end
    return m
end

function var(d::UnivariateMixtureDistribution)
    K = ncomponents(d)
    p = probs(d)
    means = Vector{Float64}(undef, K)
    m = 0.0
    v = 0.0
    for i = 1:K
        pi = p[i]
        if pi > 0.0
            ci = component(d, i)
            means[i] = mi = mean(ci)
            m += pi * mi
            v += pi * var(ci)
        end
    end
    for i = 1:K
        pi = p[i]
        if pi > 0.0
            v += pi * abs2(means[i] - m)
        end
    end
    return v
end

function cov(d::MultivariateMixtureDistribution)
    K = ncomponents(d)
    p = probs(d)
    m = zeros(length(d))
    md = zeros(length(d))
    V = zeros(length(d),length(d))

    for i = 1:K
        pi = p[i]
        if pi > 0.0
            c = component(d, i)
            BLAS.axpy!(pi, mean(c), m)
            BLAS.axpy!(pi, cov(c), V)
        end
    end
    for i = 1:K
        pi = p[i]
        if pi > 0.0
            c = component(d, i)
            # todo: use more in-place operations
            md = mean(c) - m
            BLAS.axpy!(pi, md*md', V)
        end
    end
    return V
end

#### show

function show(io::IO, d::D) where D <: AbstractMixtureDistribution
    K = ncomponents(d)
    pr = probs(d)
    println(io, "$D(K = $K)")
    Ks = min(K, 8)
    for i = 1:Ks
        @printf(io, "components[%d] (prior = %.4f): ", i, pr[i])
        println(io, component(d, i))
    end
    if Ks < K
        println(io, "The rest are omitted ...")
    end
end

"""

  A SpikeSlab distribution contains a Dirac delta and a continuous distribution.
"""
struct SpikeSlab{N<:Number,
                 C<:Distribution{Univariate, ContinuousSupport{N}}} <:
                     AbstractMixtureDistribution{Univariate,
                                                 DiscontinuousSupport{N, N}}
    pSpike::Float64
    spike::N
    slab::C
end

ncomponents(d::SpikeSlab) = 2
components(ss::SpikeSlab) = [Dirac(ss.spike), ss.slab]
component(ss::SpikeSlab, k::Integer) = k == 1 ? Dirac(ss.spike) :
    (k==2 ? ss.slab : throw(BoundsError()))
probs(ss::SpikeSlab) = [ss.pSpike, one(ss.pSpike) - ss.pSpike]
minimum(ss::SpikeSlab) = min(ss.spike, minimum(ss.slab))
maximum(ss::SpikeSlab) = max(ss.spike, maximum(ss.slab))
mean(ss::SpikeSlab) = ss.spike * ss.pSpike +
    mean(ss.slab) * (one(ss.pSpike) - ss.pSpike)
var(ss::SpikeSlab) =
    (one(ss.pSpike) - ss.pSpike) * (var(ss.slab) + mean(ss.slab)^2) +
    ss.pSpike * ss.spike^2 - mean(ss)^2
# Probability mass always beats density!
mode(ss::SpikeSlab) = iszero(ss.pSpike) ? mode(ss.slab) : ss.spike
pdf(ss::SpikeSlab, x) = (x ≈ ss.spike) ? Inf : pdf(ss.slab, x)
pmf(ss::SpikeSlab, x) =
    (iszero(ss.pSpike) || x ≉ ss.spike) ? 0.0 : ss.pSpike
logpmf(ss::SpikeSlab, x) =
    iszero(ss.pSpike) || x ≉ ss.spike ? -Inf : log(ss.pSpike)
logpdf(ss::SpikeSlab, x) = (x ≈ ss.spike) ? Inf : logpdf(ss.slab, x)
cdf(ss::SpikeSlab, x) =
    cdf(ss.slab, x) + (x < ss.spike ? zero(ss.pSpike) : ss.pSpike)
rand(rng::AbstractRNG, ss::SpikeSlab) =
    rand(rng) < ss.sSpike ? ss.spike : rand(rng, ss.slab)
function quantile(ss::SpikeSlab, p::Real)
    toSpike = cdf(ss.slab, ss.spike) * (one(ss.pSpike) - ss.pSpike)
    if p ≤ toSpike
        quantile(ss.slab, p / (one(ss.pSpike) - ss.pSpike))
    elseif p ≤ toSpike + ss.pSpike
        ss.spike
    else
        quantile(ss.slab, (p - ss.pSpike) / (one(ss.pSpike) - ss.pSpike))
    end
end
insupport(ss::SpikeSlab, x::Number) = insupport(ss.slab, x) || x ≈ ss.spike

"""

  A `CompoundDistribution` contains multiple discrete and continuous distributions.
"""
struct CompoundDistribution{TR<:Tuple, TD<:Tuple, NR<:Number, ND<:Number,
                            Counts<:Union{CountableUnivariateDistribution{ND},
                                          Nothing}} <:
    AbstractMixtureDistribution{Univariate,
                                UnionSupport{NR, ND,
                                             ContinuousSupport{NR},
                                             CountableSupport{ND}}}
    choice::Categorical{Float64, Vector{Float64}, Int}
    countable::Counts
    nContinuous::Int
    pContinuous::Float64
    continuous::TR
    discrete::TD

    function CompoundDistribution{TR, Tuple{}}(ps::Vector{Float64},
                                               continuous::TR, ::Tuple{},
                                               ::NoArgCheck) where TR
        return new{TR, Tuple{}, eltype(first(TR)), eltype(first(TR)),
                   Nothing}(Categorical(ps), nothing,
                            length(continuous), 1.0, continuous, ())
    end

    function CompoundDistribution{TR, Tuple{Counts}}(ps::Vector{Float64},
                                                     continuous::TR,
                                                     discrete::Tuple{Counts},
                                                     ::NoArgCheck) where
        {TR, Counts}
        n = length(continuous)
        return new{TR, Tuple{Counts}, eltype(first(TR)), Int,
                   Counts}(Categorical(ps), first(discrete),
                           n, sum(probs(choice)[1:n]),
                           continuous, discrete)
    end
        
    function CompoundDistribution{TR, TD}(ps::Vector{Float64},
                                          continuous::TR,
                                          discrete::TD,
                                          ::NoArgCheck) where {TR, TD}
        n = length(continuous)
        dict = Dict{ND, Float64}()
        for d in discrete
            for s in support(d)
                dict[s] = get(dict, s, 0.0)
            end
        end
        choice = DiscreteNonParametric(dict)
        DR = length(continuous) == 0 ? Float64 : eltype(first(TR))
        return new{TR, TD, DR, eltype(first(TD)),
                   typeof(choice)}(Categorical(ps), choice,
                                   n, sum(probs(choice)[1:n]),
                                   continuous, discrete)
    end

    function CompoundDistribution{TR, TD}(ps::Vector{Float64},
                                          continuous::TR,
                                          discrete::TD) where {TR <: Tuple,
                                                               TD <: Tuple}
        @check_args(CompoundDistribution,
                    length(support(choice)) ==
                    length(continuous) + length(discrete))
        if length(continuous) > 0
            vs = Set(value_support.(continuous))
            @check_args(CompoundDistribution, length(vs) == 1)
            @check_args(CompoundDistribution, eltype(first(vs)) ≡ ND)
            @check_args(CompoundDistribution,
                        length(Set(variate_form.(continuous))) == 1)
        end
        if length(discrete) > 0
            @check_args(CompoundDistribution, all(hasfinitesupport.(discrete)))
            ds = Set(map(value_support, discrete))
            @check_args(CompoundDistribution, length(ds) == 1)
            @check_args(CompoundDistribution, eltype(first(ds)) ≡ NR)
            @check_args(CompoundDistribution,
                        length(Set(variate_form.(discrete))) == 1)
        end
        return CompoundDistribution{TR, TD}(ps, continuous, discrete,
                                            NoArgCheck())
    end
end

CompoundDistribution(ps::Vector{Float64}, continuous::TR, discrete::TD) where
{TR <: Tuple, TD <: Tuple} =
    CompoundDistribution{TR, TD}(ps, continuous, discrete)
function CompoundDistribution(ps::Vector{Float64}, continuous, discrete)
    tcont = tuple(continuous...)
    tdisc = tuple(discrete...)
    return CompoundDistribution{typeof(tcont), typeof(tdisc)}(ps, tcont, tdisc)
end

function CompoundDistribution(ps::Vector{Float64}, distributions)
    continuous = filter(d -> value_support(d) <: ContinuousSupport,
                        collect(distributions))
    discrete = filter(d -> value_support(d) <: CountableSupport,
                      collect(distributions))
    return CompoundDistribution(ps, continuous, discrete)
end

CompoundDistribution(dict::Dict) = CompoundDistribution(values(dict),
                                                        keys(dict))

components(cd::CompoundDistribution) = [cd.continuous..., cd.discrete...]
probs(cd::CompoundDistribution) = probs(cd.choice)
# Probability mass always beats density!
pdf(cd::CompoundDistribution, x) =
    (cd.countable ≢ nothing && any(x .≈ support(cd.countable))) ? Inf :
    reduce(+, pdf.(cd.continuous, x)) * cd.pContinuous

pmf(cd::CompoundDistribution, x) = isnothing(cd.countable) ? 0.0 :
    (pmf(cd.countable) * (1.0 - cd.pContinuous))
function cdf(cd::CompoundDistribution, x)
    cdf_point = cdf(cd.countable, x) * (1.0 - cd.pContinuous)
    cs = cd.continuous
    ps = probs(cd.choice)
    for idx in Base.OneTo(cd.nContinuous)
        cdf_point += ps[idx] * cdf(cs[idx], x)
    end
    return cdf_point
end
rand(rng::AbstractRNG, cd::CompoundDistribution) =
    rand(rng, component(cd, rand(rng, cd.choice)))
insupport(cd::CompoundDistribution, x::Number) =
    any(insupport.(cd.continuous, x)) || any(insupport.(cd.discrete, x))
