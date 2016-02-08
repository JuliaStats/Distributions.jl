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

+(x::DenseArray, v::ZeroVector) = x
-(x::DenseArray, v::ZeroVector) = x
.+(x::DenseArray, v::ZeroVector) = x
.-(x::DenseArray, v::ZeroVector) = x


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

isprobvec(p::Vector{Float64}) = allnonneg(p) && isapprox(sum(p), 1.0)

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

add!(x::DenseArray, y::DenseVector) = broadcast!(+, x, x, y)
add!(x::DenseVecOrMat, y::ZeroVector) = x

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
