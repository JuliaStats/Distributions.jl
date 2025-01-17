"""
    LKJCholesky(d::Int, η::Real, uplo='L')

The `LKJCholesky` distribution of size ``d`` with shape parameter ``\\eta`` is a
distribution over `LinearAlgebra.Cholesky` factorisations of ``d\\times d`` real correlation
matrices (positive-definite matrices with ones on the diagonal).

Variates or samples of the distribution are `LinearAlgebra.Cholesky` objects, as might
be returned by `F = LinearAlgebra.cholesky(R)`, so that `Matrix(F) ≈ R` is a variate or
sample of [`LKJ`](@ref). 

Sampling `LKJCholesky` is faster than sampling `LKJ`, and often having the correlation
matrix in factorized form makes subsequent computations cheaper as well.

!!! note
    `LinearAlgebra.Cholesky` stores either the upper or lower Cholesky factor, related by
    `F.U == F.L'`. Both can be accessed with `F.U` and `F.L`, but if the factor
    not stored is requested, then a copy is made. The `uplo` parameter specifies whether the
    upper (`'U'`) or lower (`'L'`) Cholesky factor is stored when randomly generating
    samples. Set `uplo` to `'U'` if the upper factor is desired to avoid allocating a copy
    when calling `F.U`.

See [`LKJ`](@ref) for more details.

External links

* Lewandowski D, Kurowicka D, Joe H.
  Generating random correlation matrices based on vines and extended onion method,
  Journal of Multivariate Analysis (2009), 100(9): 1989-2001
  doi: [10.1016/j.jmva.2009.04.008](https://doi.org/10.1016/j.jmva.2009.04.008)
"""
struct LKJCholesky{T <: Real} <: Distribution{CholeskyVariate,Continuous}
    d::Int
    η::T
    uplo::Char
    logc0::T
end

#  -----------------------------------------------------------------------------
#  Constructors
#  -----------------------------------------------------------------------------

function LKJCholesky(d::Int, η::Real, _uplo::Union{Char,Symbol} = 'L'; check_args::Bool=true)
    @check_args(
        LKJCholesky,
        (d, d > 0, "matrix dimension must be positive"),
        (η, η > 0, "shape parameter must be positive"),
    )
    logc0 = lkj_logc0(d, η)
    uplo = _char_uplo(_uplo)
    T = Base.promote_eltype(η, logc0)
    return LKJCholesky(d, T(η), uplo, T(logc0))
end

# adapted from LinearAlgebra.char_uplo
function _char_uplo(uplo::Union{Symbol,Char})
    uplo ∈ (:U, 'U') && return 'U'
    uplo ∈ (:L, 'L') && return 'L'
    throw(ArgumentError("uplo argument must be either 'U' (upper) or 'L' (lower)"))
end

#  -----------------------------------------------------------------------------
#  REPL display
#  -----------------------------------------------------------------------------

Base.show(io::IO, d::LKJCholesky) = show(io, d, (:d, :η, :uplo))

#  -----------------------------------------------------------------------------
#  Conversion
#  -----------------------------------------------------------------------------

function Base.convert(::Type{LKJCholesky{T}}, d::LKJCholesky) where T <: Real
    return LKJCholesky{T}(d.d, T(d.η), d.uplo, T(d.logc0))
end
Base.convert(::Type{LKJCholesky{T}}, d::LKJCholesky{T}) where T <: Real = d

function convert(::Type{LKJCholesky{T}}, d::Integer, η::Real, uplo::Char, logc0::Real) where T <: Real
    return LKJCholesky{T}(Int(d), T(η), uplo, T(logc0))
end

#  -----------------------------------------------------------------------------
#  Properties
#  -----------------------------------------------------------------------------

Base.eltype(::Type{LKJCholesky{T}}) where {T} = T

function Base.size(d::LKJCholesky)
    p = d.d
    return (p, p)
end

function insupport(d::LKJCholesky, R::LinearAlgebra.Cholesky)
    p = d.d
    factors = R.factors
    (isreal(factors) && size(factors, 1) == p) || return false
    iinds, jinds = axes(factors)
    # check that the diagonal of U'*U or L*L' is all ones
    @inbounds if R.uplo === 'U'
        for (j, jind) in enumerate(jinds)
            col_iinds = view(iinds, 1:j)
            sum(abs2, view(factors, col_iinds, jind)) ≈ 1 || return false
        end
    else  # R.uplo === 'L'
        for (i, iind) in enumerate(iinds)
            row_jinds = view(jinds, 1:i)
            sum(abs2, view(factors, iind, row_jinds)) ≈ 1 || return false
        end
    end
    return true
end

function mode(d::LKJCholesky; check_args::Bool=true)
    @check_args(
        LKJCholesky,
        @setup(η = d.η),
        (η, η > 1, "mode is defined only when η > 1."),
    )
    factors = Matrix{eltype(d)}(LinearAlgebra.I, size(d))
    return LinearAlgebra.Cholesky(factors, d.uplo, 0)
end

StatsBase.params(d::LKJCholesky) = (d.d, d.η, d.uplo)

@inline partype(::LKJCholesky{T}) where {T <: Real} = T

#  -----------------------------------------------------------------------------
#  Evaluation
#  -----------------------------------------------------------------------------

function logkernel(d::LKJCholesky, R::LinearAlgebra.Cholesky)
    factors = R.factors
    p, η = params(d)
    c = p + 2(η - 1)
    p == 1 && return c * log(first(factors))
    # assuming D = diag(factors) with length(D) = p,
    # logp = sum(i -> (c - i) * log(D[i]), 2:p)
    logp = sum(Iterators.drop(enumerate(diagind(factors)), 1)) do (i, di) 
        return (c - i) * log(factors[di])
    end
    return logp
end

function logpdf(d::LKJCholesky, R::LinearAlgebra.Cholesky)
    lp = _logpdf(d, R)
    return insupport(d, R) ? lp : oftype(lp, -Inf)
end

_logpdf(d::LKJCholesky, R::LinearAlgebra.Cholesky) = logkernel(d, R) + d.logc0

pdf(d::LKJCholesky, R::LinearAlgebra.Cholesky) = exp(logpdf(d, R))

loglikelihood(d::LKJCholesky, R::LinearAlgebra.Cholesky) = logpdf(d, R)
function loglikelihood(d::LKJCholesky, Rs::AbstractArray{<:LinearAlgebra.Cholesky})
    return sum(R -> logpdf(d, R), Rs)
end

#  -----------------------------------------------------------------------------
#  Sampling
#  -----------------------------------------------------------------------------

function Base.rand(rng::AbstractRNG, d::LKJCholesky)
    factors = Matrix{eltype(d)}(undef, size(d))
    R = LinearAlgebra.Cholesky(factors, d.uplo, 0)
    return _lkj_cholesky_onion_sampler!(rng, d, R)
end
function Base.rand(rng::AbstractRNG, d::LKJCholesky, dims::Dims)
    p = d.d
    uplo = d.uplo
    T = eltype(d)
    TM = Matrix{T}
    Rs = Array{LinearAlgebra.Cholesky{T,TM}}(undef, dims)
    for i in eachindex(Rs)
        factors = TM(undef, p, p)
        Rs[i] = R = LinearAlgebra.Cholesky(factors, uplo, 0)
        _lkj_cholesky_onion_sampler!(rng, d, R)
    end
    return Rs
end

Random.rand!(d::LKJCholesky, R::LinearAlgebra.Cholesky) = Random.rand!(default_rng(), d, R)
function Random.rand!(rng::AbstractRNG, d::LKJCholesky, R::LinearAlgebra.Cholesky)
    return _lkj_cholesky_onion_sampler!(rng, d, R)
end

function Random.rand!(
    rng::AbstractRNG,
    d::LKJCholesky,
    Rs::AbstractArray{<:LinearAlgebra.Cholesky{T,TM}},
    allocate::Bool,
) where {T,TM}
    p = d.d
    uplo = d.uplo
    if allocate
        for i in eachindex(Rs)
            Rs[i] = _lkj_cholesky_onion_sampler!(
                rng,
                d,
                LinearAlgebra.Cholesky(TM(undef, p, p), uplo, 0),
            )
        end
    else
        for i in eachindex(Rs)
            _lkj_cholesky_onion_sampler!(rng, d, Rs[i])
        end
    end
    return Rs
end
function Random.rand!(
    rng::AbstractRNG,
    d::LKJCholesky,
    Rs::AbstractArray{<:LinearAlgebra.Cholesky{<:Real}},
)
    allocate = any(!isassigned(Rs, i) for i in eachindex(Rs)) || any(R -> size(R, 1) != d.d, Rs)
    return Random.rand!(rng, d, Rs, allocate)
end

#
# onion method
#

function _lkj_cholesky_onion_sampler!(
    rng::AbstractRNG,
    d::LKJCholesky,
    R::LinearAlgebra.Cholesky,
)
    if R.uplo === 'U'
        _lkj_cholesky_onion_tri!(rng, R.factors, d.d, d.η, Val(:U))
    else
        _lkj_cholesky_onion_tri!(rng, R.factors, d.d, d.η, Val(:L))
    end
    return R
end

function _lkj_cholesky_onion_tri!(
    rng::AbstractRNG,
    A::AbstractMatrix,
    d::Int,
    η::Real,
    ::Val{uplo},
) where {uplo}
    # Section 3.2 in LKJ (2009 JMA)
    # reformulated to incrementally construct Cholesky factor as mentioned in Section 5
    # equivalent steps in algorithm in reference are marked.
    @assert size(A) == (d, d)
    A[1, 1] = 1
    d > 1 || return A
    β = η + (d - 2)//2
    #  1. Initialization
    w0 = 2 * rand(rng, Beta(β, β)) - 1
    @inbounds if uplo === :L
        A[2, 1] = w0
    else
        A[1, 2] = w0
    end
    @inbounds A[2, 2] = sqrt(1 - w0^2)
    #  2. Loop, each iteration k adds row/column k+1
    for k in 2:(d - 1)
        #  (a)
        β -= 1//2
        #  (b)
        y = rand(rng, Beta(k//2, β))
        #  (c)-(e)
        # w is directionally uniform vector of length √y
        @inbounds w = @views uplo === :L ? A[k + 1, 1:k] : A[1:k, k + 1]
        Random.randn!(rng, w)
        rmul!(w, sqrt(y) / norm(w))
        # normalize so new row/column has unit norm
        @inbounds A[k + 1, k + 1] = sqrt(1 - y)
    end
    #  3.
    return A
end
