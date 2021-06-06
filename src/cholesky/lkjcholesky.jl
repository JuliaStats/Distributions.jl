struct LKJCholesky{T <: Real, D <: Integer} <: Distribution{CholeskyVariate,Continuous}
    d::D
    η::T
    uplo::Char
    logc0::T
end

#  -----------------------------------------------------------------------------
#  Constructors
#  -----------------------------------------------------------------------------

function LKJCholesky(d::Integer, η::Real, _uplo::Union{Char,Symbol} = 'L'; check_args = true)
    if check_args
        d > 0 || throw(ArgumentError("Matrix dimension must be positive."))
        η > 0 || throw(ArgumentError("Shape parameter must be positive."))
    end
    logc0 = lkj_logc0(d, η)
    uplo = _as_char(_uplo)
    uplo ∈ ('U', 'L') || throw(ArgumentError("uplo must be 'U' or 'L'."))
    T = Base.promote_eltype(η, logc0)
    LKJCholesky{T, typeof(d)}(d, T(η), uplo, T(logc0))
end

_as_char(c::Char) = c
_as_char(x) = only(string(x))

#  -----------------------------------------------------------------------------
#  Properties
#  -----------------------------------------------------------------------------

dim(d::LKJCholesky) = d.d

function size(d::LKJCholesky)
    p = dim(d)
    return (p, p)
end

params(d::LKJCholesky) = (d.d, d.η, d.uplo)

@inline partype(::LKJCholesky{T}) where {T <: Real} = T

#  -----------------------------------------------------------------------------
#  Evaluation
#  -----------------------------------------------------------------------------

function logkernel(d::LKJCholesky, x::Cholesky)
    factors = x.factors
    p, η = params(d)
    c = p + 2(η - 1)
    T = typeof(one(c) * log(one(eltype(factors))))
    logp = zero(T)
    di = diagind(factors)
    for i in 2:p
        logp += (c - i) * log(factors[di[i]])
    end
    return logp
end

logpdf(d::LKJCholesky, x::Cholesky) = logkernel(d, x) + d.logc0

