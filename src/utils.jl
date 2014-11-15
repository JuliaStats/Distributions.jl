
## a type to indicate zero vector

immutable ZeroVector{T} 
    len::Int 
end

ZeroVector{T}(::Type{T}, n::Int) = ZeroVector{T}(n)

eltype{T}(v::ZeroVector{T}) = T
length(v::ZeroVector) = v.len
full(v::ZeroVector) = zeros(T, v.len)

convert{T}(::Type{Vector{T}}, v::ZeroVector{T}) = full(v)

+ (x::DenseArray, v::ZeroVector) = x
- (x::DenseArray, v::ZeroVector) = x
.+ (x::DenseArray, v::ZeroVector) = x
.- (x::DenseArray, v::ZeroVector) = x


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
        if !(x == zero(T))
            return false
        end
    end
    return true
end

allzeros(x::ZeroVector) = true

function isprobvec(p::Vector{Float64})
    s = 0.
    for i = 1:length(p)
        pi = p[i]
        s += pi
        if pi < 0
            return false
        end
    end      
    return abs(s - 1.0) <= 1.0e-12
end

function pnormalize!{T<:FloatingPoint}(v::AbstractVector{T})
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

# macros for generating functions for support handling
#
# Both lb & ub should be compile-time constants
# otherwise, one should manually specify the methods
#

macro continuous_distr_support(D, lb, ub)
    if isfinite(eval(lb)) && isfinite(eval(ub))  # [lb, ub]
        esc(quote
            isupperbounded(::Union($D, Type{$D})) = true
            islowerbounded(::Union($D, Type{$D})) = true
            isbounded(::Union($D, Type{$D})) = true
            minimum(::Union($D, Type{$D})) = $lb
            maximum(::Union($D, Type{$D})) = $ub
            insupport(::Union($D, Type{$D}), x::Real) = ($lb <= x <= $ub)
        end)

    elseif isfinite(eval(lb))  # [lb, inf)
        esc(quote
            isupperbounded(::Union($D, Type{$D})) = false
            islowerbounded(::Union($D, Type{$D})) = true
            isbounded(::Union($D, Type{$D})) = false
            minimum(::Union($D, Type{$D})) = $lb
            maximum(::Union($D, Type{$D})) = $ub
            insupport(::Union($D, Type{$D}), x::Real) = (isfinite(x) && x >= $lb)
        end)

    elseif isfinite(eval(ub))  # (-inf, ub]
        esc(quote
            isupperbounded(::Union($D, Type{$D})) = true
            islowerbounded(::Union($D, Type{$D})) = false
            isbounded(::Union($D, Type{$D})) = false
            minimum(::Union($D, Type{$D})) = $lb
            maximum(::Union($D, Type{$D})) = $ub
            insupport(::Union($D, Type{$D}), x::Real) = (isfinite(x) && x <= $ub)
        end)

    else   # (-inf, inf)
        esc(quote
            isupperbounded(::Union($D, Type{$D})) = false
            islowerbounded(::Union($D, Type{$D})) = false
            isbounded(::Union($D, Type{$D})) = false
            minimum(::Union($D, Type{$D})) = $lb
            maximum(::Union($D, Type{$D})) = $ub
            insupport(::Union($D, Type{$D}), x::Real) = isfinite(x)
        end)

    end
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




