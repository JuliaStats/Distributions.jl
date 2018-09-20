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

struct ZeroVector{T}
    len::Int
end

ZeroVector(::Type{T}, n::Int) where {T} = ZeroVector{T}(n)

eltype(v::ZeroVector{T}) where {T} = T
length(v::ZeroVector) = v.len

Base.Vector(v::ZeroVector{T}) where {T} = zeros(T, v.len)
convert(::Type{Vector{T}}, v::ZeroVector{T}) where {T} = Vector(v)
convert(::Type{<:Vector}, v::ZeroVector{T}) where {T} = Vector(v)

convert(::Type{ZeroVector{T}}, v::ZeroVector) where {T} = ZeroVector{T}(length(v))

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


##### Utility functions

mutable struct NoArgCheck end

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

function pnormalize!(v::AbstractVector{T}) where T<:AbstractFloat
    s = 0.
    n = length(v)
    for i = 1:n
        @inbounds s += v[i]
    end
    for i = 1:n
        @inbounds v[i] /= s
    end
    v
end

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
