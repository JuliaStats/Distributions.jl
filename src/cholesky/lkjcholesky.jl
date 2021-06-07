struct LKJCholesky{T <: Real, D <: Integer} <: Distribution{CholeskyVariate,Continuous}
    d::D
    η::T
    uplo::Char
    logc0::T
end

#  -----------------------------------------------------------------------------
#  Constructors
#  -----------------------------------------------------------------------------

function LKJCholesky(d::Integer, _η::Real, _uplo::Union{Char,Symbol} = 'L'; check_args = true)
    if check_args
        d > 0 || throw(ArgumentError("Matrix dimension must be positive."))
        _η > 0 || throw(ArgumentError("Shape parameter must be positive."))
    end
    _logc0 = lkj_logc0(d, _η)
    uplo = _char_uplo(_uplo)
    uplo ∈ ('U', 'L') || throw(ArgumentError("uplo must be 'U' or 'L'."))
    η, logc0 = promote(_η, _logc0)
    return LKJCholesky(d, η, uplo, logc0)
end

# adapted from LinearAlgebra.char_uplo
function _char_uplo(_uplo::Union{Symbol,Char})
    uplo = if _uplo === :U
        'U'
    elseif _uplo === :L
        'L'
    else
        _uplo
    end
    uplo ∈ ('U', 'L') && return uplo
    throw(ArgumentError("uplo argument must be either 'U' (upper) or 'L' (lower)"))
end

#  -----------------------------------------------------------------------------
#  REPL display
#  -----------------------------------------------------------------------------

show(io::IO, d::LKJCholesky) = show(io, d, (:d, :η, :uplo))

#  -----------------------------------------------------------------------------
#  Conversion
#  -----------------------------------------------------------------------------

function convert(::Type{LKJCholesky{T}}, d::LKJCholesky) where T <: Real
    return LKJCholesky{T, typeof(d.d)}(d.d, T(d.η), d.uplo, T(d.logc0))
end
function convert(::Type{LKJCholesky{T}}, d::Integer, η, logc0) where T <: Real
    return LKJCholesky{T, typeof(d)}(d, T(η), d.uplo, T(logc0))
end

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
    # check that the diagonal of U'*U or L*L' is all ones
    @inbounds if R.uplo === 'U'
        for (j, jind) in enumerate(jinds)
            col_iinds = view(iinds, 1:j)
            sum(abs2(factors[iind, jind]) for iind in col_iinds) ≈ 1 || return false
        end
    else  # R.uplo === 'L'
        for (i, iind) in enumerate(iinds)
            row_jinds = view(jinds, 1:i)
            sum(abs2(factors[iind, jind]) for jind in row_jinds) ≈ 1 || return false
        end
    end
    return true
end

function mode(d::LKJCholesky)
    factors = Matrix{eltype(d)}(I, size(d))
    return Cholesky(factors, d.uplo, 0)
end

params(d::LKJCholesky) = (d.d, d.η, d.uplo)

@inline partype(::LKJCholesky{T}) where {T <: Real} = T

#  -----------------------------------------------------------------------------
#  Evaluation
#  -----------------------------------------------------------------------------

function logkernel(d::LKJCholesky, R::Cholesky)
    factors = R.factors
    p, η = params(d)
    c = p + 2(η - 1)
    logp = sum(Iterators.drop(enumerate(diagind(factors)), 1)) do (i, di) 
        return (c - i) * log(factors[di])
    end
    return logp
end

logpdf(d::LKJCholesky, R::Cholesky) = logkernel(d, R) + d.logc0

pdf(d::LKJCholesky, R::Cholesky) = exp(logpdf(d, R))

loglikelihood(d::LKJCholesky, R::Cholesky) = logpdf(d, R)
function loglikelihood(d::LKJCholesky, Rs::AbstractArray{<:Cholesky})
    return sum(R -> logpdf(d, R), Rs)
end

#  -----------------------------------------------------------------------------
#  Sampling
#  -----------------------------------------------------------------------------

function rand(rng::AbstractRNG, d::LKJCholesky)
    factors = Matrix{eltype(d)}(undef, size(d))
    R = Cholesky(factors, d.uplo, 0)
    return _lkj_cholesky_vine_sampler!(rng, d, R)
end
function rand(rng::AbstractRNG, d::LKJCholesky, dims::Dims)
    p = dim(d)
    uplo = d.uplo
    T = eltype(d)
    TM = Matrix{T}
    Rs = Array{Cholesky{T,TM}}(undef, dims)
    for i in eachindex(Rs)
        factors = TM(undef, p, p)
        Rs[i] = R = Cholesky(factors, uplo, 0)
        _lkj_cholesky_vine_sampler!(rng, d, R)
    end
    return Rs
end

rand!(rng::AbstractRNG, d::LKJCholesky, R::Cholesky) = _lkj_cholesky_vine_sampler!(rng, d, R)

function _lkj_cholesky_vine_sampler!(rng::AbstractRNG, d::LKJCholesky, R::Cholesky)
    p, η = params(d)
    factors = R.factors
    if R.uplo === 'U'
        z = _lkj_vine_rand_cpcs!(rng, transpose(factors), p, η)
        _cpcs_to_cholesky!(factors, z, p)
    else  # uplo === 'L'
        z = _lkj_vine_rand_cpcs!(rng, factors, p, η)
        _cpcs_to_cholesky!(transpose(factors), z, p)
    end
    return R
end

# sample partial canonical correlations z using the vine method from Section 2.4 in LKJ (2009 JMA)
function _lkj_vine_rand_cpcs!(rng::AbstractRNG, z::AbstractMatrix, d::Integer, η::Real)
    β = η + (d - 1) // 2
    for i in 1:(d - 1)
        β -= 1 // 2
        spl = sampler(Beta(β, β))
        # use function barrier since sampler(Beta(β, β)) is not type-stable
        # see https://github.com/JuliaStats/Distributions.jl/pull/1281#issue-573169822
        _lkj_vine_rand_cpcs_loop!(rng, spl, z, i, (i + 1):d)
    end
    return z
end

function _lkj_vine_rand_cpcs_loop!(rng, sampler, z, i, js)
    @inbounds for j in js
        z[i, j] = 2 * rand(rng, sampler) - 1
    end
    return z
end

# map partial canonical correlations z stored in the upper triangle of a matrix
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
