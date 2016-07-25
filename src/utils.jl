## macro for argument checking

macro check_args(D, cond)
    quote
        if !($cond)
            throw(ArgumentError(string(
                $(string(D)), ": the condition ", $(string(cond)), " is not satisfied.")))
        end
    end
end

## a type to indicate zero vector

immutable ZeroVector{T}
    len::Int
end

ZeroVector{T}(::Type{T}, n::Int) = ZeroVector{T}(n)

eltype{T}(v::ZeroVector{T}) = T
length(v::ZeroVector) = v.len
full{T}(v::ZeroVector{T}) = zeros(T, v.len)

convert{T}(::Type{Vector{T}}, v::ZeroVector{T}) = full(v)

+(x::AbstractArray, v::ZeroVector) = x
-(x::AbstractArray, v::ZeroVector) = x
.+(x::AbstractArray, v::ZeroVector) = x
.-(x::AbstractArray, v::ZeroVector) = x


##### Utility functions

type NoArgCheck end

isunitvec{T}(v::AbstractVector{T}) = (vecnorm(v) - 1.0) < 1.0e-12

function allfinite{T<:Real}(x::Array{T})
    for i = 1 : length(x)
        if !(isfinite(x[i]))
            return false
        end
    end
    return true
end

function allzeros{T<:Real}(x::Array{T})
    for i = 1 : length(x)
        if !(x[i] == zero(T))
            return false
        end
    end
    return true
end

allzeros(x::ZeroVector) = true

function allnonneg{T<:Real}(x::Array{T})
    for i = 1 : length(x)
        if !(x[i] >= zero(T))
            return false
        end
    end
    return true
end

isprobvec{T<:Real}(p::Vector{T}) = allnonneg(p) && isapprox(sum(p), one(T))

function pnormalize!{T<:AbstractFloat}(v::AbstractVector{T})
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
    :(zero($p) <= $p <= one($p) ? $ex : NaN)
end
macro checkinvlogcdf(lp,ex)
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
hasCholesky(a::Matrix{Float64}) = isa(trycholfact(a), Cholesky)

function trycholfact(a::Matrix{Float64})
    try cholfact(a)
    catch e
        return e
    end
end

# for when container inputs need to be promoted to the same eltype
function promote_eltype{T, S}(A::Array{T}, B::Array{S})
    R = promote_type(T, S)
    (Array{R}(A), Array{R}(B))
end
function promote_eltype{T}(A::Array{T}, B::Real)
    R = promote_type(T, typeof(B))
    (Array{R}(A), R(B))
end
function promote_eltype(A::Real, B::Array)
    tup = promote_eltype(B, A)
    (tup[2], tup[1])
end
function promote_eltype{T, S}(A::Array{T}, B::AbstractPDMat{S})
    R = promote_type(T, S)
    (Array{R}(A), convert(typeof(B).name.primary{R}, B))
end
function promote_eltype{S}(A::Real, B::AbstractPDMat{S})
    R = promote_type(typeof(A), S)
    (R(A), convert(typeof(B).name.primary{R}, B))
end
promote_eltype{T, S}(A::ZeroVector{T}, B::AbstractPDMat{S}) = (ZeroVector{S}(A.len), B)
