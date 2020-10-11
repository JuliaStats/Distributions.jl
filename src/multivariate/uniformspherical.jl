# Uniform distribution on the n-sphere in R^{n+1}
#
# The implementation here follows:
#
#   - Wikipedia:
#     https://en.wikipedia.org/wiki/N-sphere

struct UniformSpherical{T<:Real} <: ContinuousMultivariateDistribution
    n::Int
    logS::T  # log normalization constant

    function UniformSpherical{T}(n::Int) where {T<:Real}
        n >= 0 || error("n must be non-negative")

        ln2 = log(2)
        if n > 0
            p = (n+1) / 2.0
            logS = ln2 + p*(log2π - ln2) - loggamma(p)
        else
            logS = ln2
        end
        new{T}(n, convert(T, logS))
    end
end
UniformSpherical(n::Int) = UniformSpherical{Float64}(n)

show(io::IO, d::UniformSpherical) = show(io, d, (:n,))

eltype(d::UniformSpherical{T}) where T = Vector{T}

### Conversions
convert(::Type{UniformSpherical{T}}, d::UniformSpherical) where {T<:Real} = UniformSpherical{T}(d.n)


### Basic properties

length(d::UniformSpherical) = d.n + 1

mean(d::UniformSpherical{T}) where T = zeros(T, length(d))
meandir(d::UniformSpherical) = mean(d)  # should this error out instead?

cov(d::UniformSpherical{T}) where T = Diagonal{T}(var(d))
var(d::UniformSpherical{T}) where T = ones(T, length(d)) / length(d)

concentration(d::UniformSpherical{T}) where T = zero(T)

insupport(d::UniformSpherical, x::AbstractVector{T}) where {T<:Real} = isunitvec(x)
params(d::UniformSpherical) = (d.n,)
# @inline partype(d::UniformSpherical{T}) where {T<:Real} = T

### Evaluation

_logpdf(d::UniformSpherical, x::AbstractVector{T}) where {T<:Real} = insupport(d, x) ? -d.logS : -T(Inf)

entropy(d::UniformSpherical) = d.logS


### Sampling

sampler(d::UniformSpherical{T}) where T = UniformSphericalSampler(d.n)


for A in [:AbstractVector, :AbstractMatrix]
    @eval function _rand!(rng::AbstractRNG, d::UniformSpherical, x::$A)
        if length(d) == 1
            # in 1D, reduces to a U{-1, 1}
            for i in eachindex(x)
                @inbounds x[i] = rand(rng, (-1, 1))
            end
        else
            _rand!(rng, sampler(d), x)
        end
        return x
    end
end


### Estimation
fit_mle(::Type{<:UniformSpherical}, X::Matrix{T}) where {T <: Real} = UniformSpherical{T}(size(X, 1) - 1)
