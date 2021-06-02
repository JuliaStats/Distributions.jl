"""
    DiscreteNormal{S<:Real, M<:Real}

Represents the (truncated) discrete normal distribution.

For an integer `x` in the support, the probability is proportional to

    P(x; µ, σ) ∝ ℯ^(-(x-μ)^2/2σ^2)

The mean and standard deviation may be any real type, not necessarily integer.
The type of σ determines the precision of the approximation of the distribution
to the true discrete Gaussian. The exact relationship depends on the sampler
used. See [MW19] for details.

If the tail cut it not explicitly provided, it is chosen depending on the
precision of σ such that the probability mass of any tail element is equal to
0.0 when properly rounded in the given floating point type (note that this does
not mean that the probability mass of the clipped tail is necessarily 0.0).
Note that this means that the tail cut need to be explicitly provided if σ is
given in an arbitrary precision type.

[MW19] Michael Walter, "Sampling the Integers with Low Relative Error"
       (https://eprint.iacr.org/2019/068).
"""
struct DiscreteNormal{S<:Real, M<:Real} <: DiscreteUnivariateDistribution
    μ::M
    σ::S
    tail_cut::UnitRange{Int}
end
const DiscreteGaussian = DiscreteNormal
minimum(d::DiscreteNormal) = first(d.tail_cut)
maximum(d::DiscreteNormal) = last(d.tail_cut)

function default_half_width(σ::T) where {T<:AbstractFloat}
    # Once the probability is below 1/2 nextfloat(zero(T)), the properly
    # rounded value of the probability is zero. Our samplers give
    # guarantees on the distance between the sampled distribution and the
    # underlying distribution, assuming probabilities are properly rounded.
    # Thus, the discrete gaussian is properly bounded through underflow of
    # the corresponding floating point format.
    fexp = -exponent(nextfloat(zero(T)))
    σ * √(2 * log(2) * (1 + fexp))
end

function DiscreteNormal(μ::M, σ::S) where {M, S}
    min = ceil(Int, μ - default_half_width(σ))
    max = floor(Int, μ + default_half_width(σ))
    DiscreteNormal(μ, σ, min:max)
end

# DiscreteNormal from generic samplers.
struct DiscreteNormalGeneric{T, D<:DiscreteNormal{T}, S} <: Sampleable{Univariate,Discrete}
    dist::D
    sampler::S
end

function DiscreteNormalGeneric(d::DiscreteNormal, S::Type)
    # Compute the probabilities for each value
    probs = [exp(-(x-d.μ)^2/(2d.σ^2)) for x = minimum(d):maximum(d)]
    probs ./= sum(probs)
    DiscreteNormalGeneric(d, S(probs))
end

function rand(rng::AbstractRNG, generic::DiscreteNormalGeneric)
    # The generic samplers sample the index space of the probability vector
    # (i.e. 1:n). Since we provided the probabilities as a minimum(d):maximum(d)
    # vector, we simply need to everything.
    rand(rng, generic.sampler) + minimum(generic.dist) - 1
end

# For a single sample - always use Karney's algorithm - It'll be faster than
# precomputing the inversion tables.
rand(rng::AbstractRNG, d::DiscreteNormal) = rand(rng, DiscreteNormalKarney(d))

function sampler(rng::AbstractRNG, d::DiscreteNormal)
    # Use a polyalgorithm. The tradeoff is a bit tricky because the primary cost
    # of the alias table algorithm is the memory requirement, which Karney's
    # algorithm doesn't have.
    if d.σ > 100 # TODO: Determine the right value experimentally
        return DiscreteNormalKarney(d)
    else
        return DiscreteNormalGeneric(d, AliasTable)
    end
end


"""
An implementation of Karney's algorithm for sampling discrete gaussians
from [CFFK13], which modifications for floating point numbers from [MW19].

[CFFK13] Charles F. F. Karney "Sampling exactly from the normal distribution".
        https://arxiv.org/abs/1303.6257
"""
module Karney

using Distributions
using Distributions: DiscreteNormal
using Random
import Random: rand

export DiscreteNormalKarney

struct DiscreteNormalKarney{T,S} <: Sampleable{Univariate,Discrete}
    d::DiscreteNormal{T,S}
end

const exphalf = exp(-1/2)

"""
    Performs repeated bernoulli trials with probability ℯ^(-½), counting the
    number of trials until the first failure. This result in an integer `k`
    being chosen with probability (1-1/√ℯ)ℯ^(-½k) (equivalently
    ℯ^(-½k) * (1 - ℯ^(-½)).
"""
function sample_k(rng)
    # Step D1
    k = 0
    # N.B.: For the non-approximated version, we could use Algorithm H from
    # [CFFK13] here.
    while rand(rng, Bernoulli(exphalf))
        k += 1
    end
    return k
end

function sample_ksj(rng, d)
    k = sample_k(rng)

    # Step D2.
    # TODO: Other implementations perform this computation using
    # repeated Bernoulli trials of exp(-1/2). I am not sure why this is.
    # Accuracy wise, we should get correctly rounded values over the
    # entire support of the discrete gaussian. Perhaps this rounding is
    # still too large?
    # Performance wise, a Bernoulli trial is about 10x as expensive as
    if k >= 2 && !rand(rng, Bernoulli(exp(-1/2*k*(k-1))))
        return nothing
    end
    positive = rand(rng, Bool)
    j = rand(rng, DiscreteUniform(0, ceil(Int64, d.σ) - 1))

    # Step D6 (c.f. CheckRejectB from [MW19])
    if !positive && iszero(d.μ) && k == 0 && j == 0
        return nothing
    end

    return (k, positive, j)
end

# Generic Karnery implementation
function rand(rng::AbstractRNG, d::DiscreteNormalKarney)
    while true
        tup = sample_ksj(rng, d.d)
        tup === nothing && continue
        (k, positive, j) = tup
        (σ, μ) = (d.d.σ, d.d.μ)

        # Step D4
        sμ = positive ? μ : -μ
        di₀ = σ * k + sμ
        i₀ = ceil(Int64, di₀)
        x₀ = i₀ - di₀
        x = (x₀ + j)/σ

        # Step D5
        if !isinteger(σ) && x >= 1
            continue
        end

        @assert x !== 0 || k !== 0
        # Step D7
        if !rand(rng, Bernoulli(exp(-1/2*x*(2k+x))))
            continue
        end

        # Step D8
        i = i₀ + j
        positive || (i = -i)

        i in d.d.tail_cut || continue

        return i
    end
end

%₁(f) = modf(f)[1]

function isx̄zero(σ, μ, k, positive)
    b = %₁(k*σ)
    if positive
        return ((b == 0 && iszero(μ)) || (b + μ) == 1)
    else
        return b == μ
    end
end

# Checks (exactly) if kσ + positive ? μ : -μ >= 1
function check_reject_a(σ, μ, k, positive, j)
    if j < floor(Int, σ) || isx̄zero(σ, μ, k, positive)
        return false
    end
    b = %₁(k*σ)
    a = %₁(σ)
    z = a+b
    if positive
        return z + μ <= (b + μ >= 1 ? 2 : 1)
    elseif b > μ
        return z <= 1 || μ <= %₁(z)
    else
        return z >= μ
    end
end

function compute_i(σ, μ, k, positive)
    i = floor(Int, k*σ)
    b = %₁(k*σ)
    if !positive && b > μ
        i += 1
    end
    if positive && (b > 0 || μ > 0)
        if b + μ <= 1
            i += 1
        else
            i += 2
        end
    end
    return i
end

function compute_x(σ, μ, k, positive, j)
    if isx̄zero(σ, μ, k, positive)
        return j/σ
    end
    b = %₁(k*σ)
    x̄ = 1 - (b + (positive ? μ : -μ))
    if positive
        (b + μ) >= 1 && (x̄ += 1)
    else
        b < μ && (x̄ -= 1)
    end
    return (x̄ + j)/σ
end

# KarneyFP from [MW19]
function rand(rng::AbstractRNG, d::DiscreteNormalKarney{<:AbstractFloat})
    while true
        tup = sample_ksj(rng, d.d)
        tup === nothing && continue
        (k, positive, j) = tup
        (σ, μ) = (d.d.σ, d.d.μ)

        # Step D5
        if !isinteger(σ) && check_reject_a(σ, μ, k, positive, j)
            continue
        end

        i₀ = compute_i(σ, μ, k, positive)
        x = compute_x(σ, μ, k, positive, j)
        if !rand(rng, Bernoulli(exp(-1/2*x*(2k+x))))
            continue
        end

        # Step D8
        i = i₀ + j
        positive || (i = -i)

        return i
    end
end

end
using .Karney
