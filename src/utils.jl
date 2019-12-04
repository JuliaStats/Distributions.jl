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
