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

function insupport(d::LKJCholesky, R::Cholesky)
    p = dim(d)
    factors = R.factors
    (isreal(factors) && size(factors, 1) == p) || return false
    iinds, jinds = axes(factors)
    T = typeof(one(eltype(R))^2)
    # check that the diagonal of U'*U or L*L' is all ones
    @inbounds if R.uplo === 'U'
        for j in 1:p
            s = zero(T)
            for i in 1:j
                s += factors[iinds[i], jinds[j]]^2
            end
            isapprox(s, 1) || return false
        end
    else  # R.uplo === 'L'
        for i in 1:p
            s = zero(T)
            for j in 1:i
                s += factors[iinds[i], jinds[j]]^2
            end
            isapprox(s, 1) || return false
        end
    end
    return true
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

#  -----------------------------------------------------------------------------
#  Sampling
#  -----------------------------------------------------------------------------

function rand(rng::AbstractRNG, d::LKJCholesky)
    p = dim(d)
    T = eltype(d)
    factors = Matrix{T}(undef, p, p)
    R = Cholesky(factors, d.uplo, 0)
    return rand!(rng, d, R)
end
function rand(rng::AbstractRNG, d::LKJCholesky, dims::Dims)
    p = dim(d)
    uplo = d.uplo
    T = eltype(d)
    TChol = Cholesky{T,Matrix{T}}
    Rs = Array{TChol}(undef, dims)
    for i in eachindex(Rs)
        factors = Matrix{T}(undef, p, p)
        Rs[i] = R = Cholesky(factors, uplo, 0)
        rand!(rng, d, R)
    end
    return Rs
end

rand!(rng::AbstractRNG, d::LKJCholesky, R::Cholesky) = _lkj_cholesky_vine_sampler!(rng, d, R)

function _lkj_cholesky_vine_sampler!(rng::AbstractRNG, d::LKJCholesky, R::Cholesky)
    p, η = params(d)
    factors = R.factors
    if R.uplo === 'U'
        cpcs = _lkj_vine_rand_cpcs!(transpose(factors), p, η, rng)
        _cpcs_to_cholesky!(factors, cpcs, p)
    else  # uplo === 'L'
        cpcs = _lkj_vine_rand_cpcs!(factors, p, η, rng)
        _cpcs_to_cholesky!(transpose(factors), cpcs, p)
    end
    return R
end

# sample partial canonical correlations using the vine method from Section 2.4 in LKJ (2009 JMA)
# storage 
function _lkj_vine_rand_cpcs!(cpcs, d::Integer, η::Real, rng::AbstractRNG)
    T = eltype(cpcs)
    β = η + T(d - 1) / 2
    @inbounds for i in 1:(d - 1)
        β -= T(0.5)
        for j in (i + 1):d
            cpcs[i, j] = 2 * rand(rng, Beta(β, β)) - 1
        end
    end
    return cpcs
end

# map partial canonical correlations stored in the upper triangle of a matrix
# to the corresponding upper triangular cholesky factor w
# adapted from https://github.com/TuringLang/Bijectors.jl/blob/v0.9.4/src/bijectors/corr.jl
function _cpcs_to_cholesky!(w, z, d::Integer)
    @inbounds for j in 1:d
        w[1, j] = 1
        for i in 1:(j-1)
            wij = w[i, j]
            zij = z[i, j]
            w[i, j] = zij * wij
            w[i+1, j] = wij * sqrt(1 - zij^2)
        end
    end
    return w
end
