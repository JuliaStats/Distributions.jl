## macro for argument checking

macro check_args(D, cond)
    quote
        if !($(esc(cond)))
            throw(ArgumentError(string(
                $(string(D)), ": the condition ", $(string(cond)), " is not satisfied.")))
        end
    end
end

##### Utility functions

isunitvec(v::AbstractVector) = (norm(v) - 1.0) < 1.0e-12

isprobvec(p::AbstractVector{<:Real}) =
    all(x -> x ≥ zero(x), p) && isapprox(sum(p), one(eltype(p)))

pnormalize!(v::AbstractVector{<:Real}) = (v ./= sum(v); v)

add!(x::AbstractArray, y::AbstractVector) = broadcast!(+, x, x, y)
add!(x::AbstractArray, y::Zeros) = x

multiply!(x::AbstractArray, c::Number) = (x .*= c; x)

exp!(x::AbstractArray) = (x .= exp.(x); x)

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

"""
    ispossemdef(A, k) -> Bool
Test whether a matrix is positive semi-definite with specified rank `k` by
checking that `k` of its eigenvalues are positive and the rest are zero.
# Examples
```jldoctest; setup = :(using Distributions: ispossemdef)
julia> A = [1 0; 0 0]
2×2 Matrix{Int64}:
 1  0
 0  0
julia> ispossemdef(A, 1)
true
julia> ispossemdef(A, 2)
false
```
"""
function ispossemdef(X::AbstractMatrix, k::Int;
                     atol::Real=0.0,
                     rtol::Real=(minimum(size(X))*eps(real(float(one(eltype(X))))))*iszero(atol))
    _check_rank_range(k, minimum(size(X)))
    ishermitian(X) || return false
    dp, dz, dn = eigsigns(Hermitian(X), atol, rtol)
    return dn == 0 && dp == k
end
function ispossemdef(X::AbstractMatrix;
                     atol::Real=0.0,
                     rtol::Real=(minimum(size(X))*eps(real(float(one(eltype(X))))))*iszero(atol))
    ishermitian(X) || return false
    dp, dz, dn = eigsigns(Hermitian(X), atol, rtol)
    return dn == 0
end

function _check_rank_range(k::Int, n::Int)
    0 <= k <= n || throw(ArgumentError("rank must be between 0 and $(n) (inclusive)"))
    nothing
end

#  return counts of the number of positive, zero, and negative eigenvalues
function eigsigns(X::AbstractMatrix,
                  atol::Real=0.0,
                  rtol::Real=(minimum(size(X))*eps(real(float(one(eltype(X))))))*iszero(atol))
    eigs = eigvals(X)
    eigsigns(eigs, atol, rtol)
end
function eigsigns(eigs::Vector{<: Real}, atol::Real, rtol::Real)
    tol = max(atol, rtol * eigs[end])
    eigsigns(eigs, tol)
end
function eigsigns(eigs::Vector{<: Real}, tol::Real)
    dp = count(x -> tol < x, eigs)        #  number of positive eigenvalues
    dz = count(x -> -tol < x < tol, eigs) #  number of numerically zero eigenvalues
    dn = count(x -> x < -tol, eigs)       #  number of negative eigenvalues
    return dp, dz, dn
end
