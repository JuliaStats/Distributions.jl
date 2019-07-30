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

Base.broadcast(::Union{typeof(+),typeof(-)}, x::AbstractArray, v::ZeroVector) = x
Base.broadcast(::typeof(+), v::ZeroVector, x::AbstractArray) = x
Base.broadcast(::typeof(-), v::ZeroVector, x::AbstractArray) = -x

Base.broadcast(::Union{typeof(+),typeof(-)}, x::Number, v::ZeroVector) = fill(x, v.len)
Base.broadcast(::typeof(+), v::ZeroVector, x::Number) = fill(x, v.len)
Base.broadcast(::typeof(-), v::ZeroVector, x::Number) = fill(-x, v.len)
Base.broadcast(::typeof(*), v::ZeroVector, ::Number) = v

##### Utility functions

"""
    NoArgCheck

Flag structure used on distribution constructors to bypass parameter validation.
"""
struct NoArgCheck end

isunitvec(v::AbstractVector{T}) where {T} = (norm(v) - 1.0) < 1.0e-12

function allfinite(x::AbstractArray{T}) where {T<:Real}
    for i in eachindex(x)
        if !isfinite(x[i])
            return false
        end
    end
    return true
end

function allzeros(x::AbstractArray{T}) where {T<:Real}
    for i in eachindex(x)
        if !(x[i] == zero(T))
            return false
        end
    end
    return true
end

allzeros(x::ZeroVector) = true

allnonneg(xs::AbstractArray{<:Real}) = all(x -> x >= 0, xs)

isprobvec(p::AbstractVector{T}) where {T<:Real} =
    allnonneg(p) && isapprox(sum(p), one(T))

pnormalize!(v::AbstractVector{<:Real}) = (v ./= sum(v); v)

add!(x::AbstractArray, y::AbstractVector) = broadcast!(+, x, x, y)
add!(x::AbstractVecOrMat, y::ZeroVector) = x

function multiply!(x::AbstractArray, c::Number)
    for i in eachindex(x)
        @inbounds x[i] *= c
    end
    return x
end

function exp!(x::AbstractArray)
    for i in eachindex(x)
        @inbounds x[i] = exp(x[i])
    end
    return x
end

# get a type wide enough to represent all a distributions's parameters
# (if the distribution is parametric)
# if the distribution is not parametric, we need this to be a float so that
# inplace pdf calculations, etc. allocate storage correctly
@inline partype(::Distribution) = Float64

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
function isApproxSymmmetric(a::AbstractMatrix{Float64})
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
