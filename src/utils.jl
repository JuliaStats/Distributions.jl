## macro for argument checking

macro check_args(D, cond)
    quote
        if !($(esc(cond)))
            throw(ArgumentError(string(
                $(string(D)), ": the condition ", $(string(cond)), " is not satisfied.")))
        end
    end
end

## a type to indicate zero vector
"""
An immutable vector of zeros of type T
"""
struct ZeroVector{T} <: AbstractVector{T}
    len::Int
end

ZeroVector(::Type{T}, n::Int) where {T} = ZeroVector{T}(n)

Base.eltype(v::ZeroVector{T}) where {T} = T
Base.length(v::ZeroVector) = v.len
Base.size(v::ZeroVector) = (v.len,)
Base.getindex(v::ZeroVector{T}, i) where {T} = zero(T)

Base.Vector(v::ZeroVector{T}) where {T} = zeros(T, v.len)
Base.convert(::Type{Vector{T}}, v::ZeroVector{T}) where {T} = Vector(v)
Base.convert(::Type{<:Vector}, v::ZeroVector{T}) where {T} = Vector(v)

Base.convert(::Type{ZeroVector{T}}, v::ZeroVector) where {T} = ZeroVector{T}(length(v))

for T = (:AbstractArray, :Number), op = (:+, :-)
    @eval begin
        Base.@deprecate ($op)(x::$T, v::ZeroVector) broadcast($op, x, v)
        Base.@deprecate ($op)(v::ZeroVector, x::$T) broadcast($op, v, x)
    end
end

Base.broadcast(::Union{typeof(+),typeof(-)}, x::AbstractArray, v::ZeroVector) = x
Base.broadcast(::typeof(+), v::ZeroVector, x::AbstractArray) = x
Base.broadcast(::typeof(-), v::ZeroVector, x::AbstractArray) = -x

Base.broadcast(::Union{typeof(+),typeof(-)}, x::Number, v::ZeroVector) = fill(x, v.len)
Base.broadcast(::typeof(+), v::ZeroVector, x::Number) = fill(x, v.len)
Base.broadcast(::typeof(-), v::ZeroVector, x::Number) = fill(-x, v.len)
Base.broadcast(::typeof(*), v::ZeroVector, ::Number) = v

##### Utility functions

struct NoArgCheck end

isunitvec(v::AbstractVector{T}) where {T} = (norm(v) - 1.0) < 1.0e-12

function allfinite(x::Array{T}) where T<:Real
    for i = 1 : length(x)
        if !(isfinite(x[i]))
            return false
        end
    end
    return true
end

function allzeros(x::Array{T}) where T<:Real
    for i = 1 : length(x)
        if !(x[i] == zero(T))
            return false
        end
    end
    return true
end

allzeros(x::ZeroVector) = true

function allnonneg(x::Array{T}) where T<:Real
    for i = 1 : length(x)
        if !(x[i] >= zero(T))
            return false
        end
    end
    return true
end

isprobvec(p::Vector{T}) where {T<:Real} = allnonneg(p) && isapprox(sum(p), one(T))

pnormalize!(v::AbstractVector{<:Real}) = (v ./= sum(v); v)

add!(x::AbstractArray, y::AbstractVector) = broadcast!(+, x, x, y)
add!(x::AbstractVecOrMat, y::ZeroVector) = x

function multiply!(x::AbstractArray, c::Number)
    for i = 1:length(x)
        @inbounds x[i] *= c
    end
    x
end

function exp!(x::AbstractArray)
    for i = 1:length(x)
        @inbounds x[i] = exp(x[i])
    end
    x
end

# get a type wide enough to represent all a distributions's parameters
# (if the distribution is parametric)
# if the distribution is not parametric, we need this to be a float so that
# inplace pdf calculations, etc. allocate storage correctly
@inline partype(d::Distribution) = Float64

# for checking the input range of quantile functions
# comparison with NaN is always false, so no explicit check is required
macro checkquantile(p,ex)
    p, ex = esc(p), esc(ex)
    :(zero($p) <= $p <= one($p) ? $ex : NaN)
end
macro checkinvlogcdf(lp,ex)
    lp, ex = esc(lp), esc(ex)
    :($lp <= zero($lp) ? $ex : NaN)
end

# because X == X' keeps failing due to floating point nonsense
function isApproxSymmmetric(a::Matrix{Float64})
    tmp = true
    for j in 2:size(a, 1)
        for i in 1:(j - 1)
            tmp &= abs(a[i, j] - a[j, i]) < 1e-8
        end
    end
    return tmp
end

# because isposdef keeps giving the wrong answer for samples
# from Wishart and InverseWisharts
hasCholesky(a::Matrix{Float64}) = isa(trycholesky(a), Cholesky)

function trycholesky(a::Matrix{Float64})
    try cholesky(a)
    catch e
        return e
    end
end

"""
compute mean over a function(uniformly distributed probabilities)
    
used to estimate moments of logitnorm
"""
function meanFunOfProb(d::ContinuousUnivariateDistribution;relPrec = 1e-4, maxCnt=2^18, fun=(d,p)->quantile.(d,p) )
    δ = 1/32 # start with 31 points (32 intervals between 0 and 1)
    p = δ:δ:(1-δ)
    # for K=1/δ intervals, there are (K-1) points at c_i
    # The first points at δ represents interval (δ/2,3/2δ)
    # The following picture shows points and intervals for K = 4
    #---|---|---|---#
    # |---|---|---| #
    # we need to add points for δ/4 and 1-δ/4 representing the edges
    # but their weight is only half, because they represents half an inverval
    #m = sum(c_i*δ) + el*(δ/2) + er*(δ/2) = (sum(c_i) + er/2 + el/2)*δ
    s = sum(fun.(d,p))   # sum at points c_i
    el = fun(d,δ/4)  # 
    er = fun(d,1-δ/4)
    m = (s + el/2 + er/2)*δ
    relErr = 1
    while 1/δ < maxCnt
        mPrev = m
        δ  = δ / 2
        # to double the number of reference points, half the interval
        # for each second point we already computed fun
        # only need to add the new points to the sum of central points
        p = δ:δ*2:(1-δ) # points at the center of current intervals
        s += sum(fun.(d,p))
        el = fun(d,δ/4)
        er = fun(d,1-δ/4)
        m = (s + el/2 + er/2)*δ
        relErr = abs(m - mPrev)/m 
        #println("cnt=$(1/δ), m=$m, mPrev=$mPrev, relErr=$relErr")
        #if the estimate did not change much, can return
        relErr <= relPrec && break
    end
    relErr > relPrec && @warn "Returning meanFunOfProb results of low relative precision of $relErr"
    m
end

# estimate the mean by numerical integration over uniform percentiles
estimateMean(d::ContinuousUnivariateDistribution;kwargs...) = 
  meanFunOfProb(d;kwargs...,fun=(d,p)->quantile(d,p))

# estimate variance by numerical integration over uniform percentiles
function estimateVariance(d::ContinuousUnivariateDistribution; mean=missing, kwargs...)
    m = ismissing(mean) ? Distributions.mean(d) : mean
    function squaredDiff(d,p)
        t = quantile(d, p) - m
        t*t
    end
    meanFunOfProb(d;kwargs...,fun=squaredDiff)
end
