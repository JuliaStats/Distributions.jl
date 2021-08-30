# Uniform distribution on the n-ball in R^n
#
# The implementation here follows:
#
#   - Wikipedia:
#     https://en.wikipedia.org/wiki/N-sphere

struct UniformBall{T<:Real} <: ContinuousMultivariateDistribution
    n::Int
    logV::T  # log normalization constant

    function UniformBall{T}(n::Int) where {T<:Real}
        n >= 0 || error("n must be non-negative")

        if n > 0
            logV = convert(T, (n/2)*log(π) - loggamma((n/2) + 1))
        else
            logV = zero(T)
        end
        new{T}(n, logV)
    end
end
UniformBall(n::Int) = UniformBall{Float64}(n)

show(io::IO, d::UniformBall) = show(io, d, (:n,))

### Conversions
convert(::Type{UniformBall{T}}, d::UniformBall) where {T<:Real} = UniformBall{T}(d.n)


### Basic properties

length(d::UniformBall) = d.n

mean(d::UniformBall{T}) where T = zeros(T, length(d))
meandir(d::UniformBall) = mean(d)

cov(d::UniformBall{T}) where T = Diagonal{T}(var(d))
var(d::UniformBall{T}) where T = ones(T, length(d)) / (2 + length(d))

insupport(d::UniformBall, x::AbstractVector{T}) where {T<:Real} = (norm(x) <= one(T)) || isunitvec(x)
params(d::UniformBall) = (d.n,)
# @inline partype(d::UniformBall{T}) where {T<:Real} = T

### Evaluation

_logpdf(d::UniformBall, x::AbstractVector{T}) where {T<:Real} = insupport(d, x) ? -d.logV : -T(Inf)

entropy(d::UniformBall) = d.logV

### Sampling

sampler(d::UniformBall{T}) where T = UniformBallSampler(d.n)

for A in [:AbstractVector, :AbstractMatrix]
    @eval function _rand!(rng::AbstractRNG, d::UniformBall, x::$A)
        if length(d) == 1
            # in 1D, reduces to U[-1, 1]
            for i in eachindex(x)
                @inbounds x[i] = 2*rand(rng) - 1
            end
        else
            _rand!(rng, sampler(d), x)
        end
        return x
    end
end


### Estimation
fit_mle(::Type{<:UniformBall}, X::Matrix{T}) where {T <: Real} = UniformBall{T}(size(X, 1))
