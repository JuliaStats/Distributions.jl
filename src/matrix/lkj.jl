"""
    LKJ(d, η)

```julia
d::Int   dimension
η::Real  positive shape
```
The [LKJ](https://doi.org/10.1016/j.jmva.2009.04.008) distribution is a distribution over
``d\\times d`` real correlation matrices (positive-definite matrices with ones on the diagonal).
If ``\\mathbf{R}\\sim \\textrm{LKJ}_{d}(\\eta)``, then its probability density function is

```math
f(\\mathbf{R};\\eta) = \\left[\\prod_{k=1}^{d-1}\\pi^{\\frac{k}{2}}
\\frac{\\Gamma\\left(\\eta+\\frac{d-1-k}{2}\\right)}{\\Gamma\\left(\\eta+\\frac{d-1}{2}\\right)}\\right]^{-1}
|\\mathbf{R}|^{\\eta-1}.
```

If ``\\eta = 1``, then the LKJ distribution is uniform over
[the space of correlation matrices](https://www.jstor.org/stable/2684832).

!!! note
    if a Cholesky factor of the correlation matrix is desired, it is more efficient to
    use [`LKJCholesky`](@ref), which avoids factorizing the matrix.
"""
struct LKJ{T <: Real, D <: Integer} <: ContinuousMatrixDistribution
    d::D
    η::T
    logc0::T
end

#  -----------------------------------------------------------------------------
#  Constructors
#  -----------------------------------------------------------------------------

function LKJ(d::Integer, η::Real; check_args::Bool=true)
    @check_args(
        LKJ,
        (d, d > 0, "matrix dimension must be positive."),
        (η, η > 0, "shape parameter must be positive."),
    )
    logc0 = lkj_logc0(d, η)
    T = Base.promote_eltype(η, logc0)
    LKJ{T, typeof(d)}(d, T(η), T(logc0))
end

#  -----------------------------------------------------------------------------
#  REPL display
#  -----------------------------------------------------------------------------

show(io::IO, d::LKJ) = show_multline(io, d, [(:d, d.d), (:η, d.η)])

#  -----------------------------------------------------------------------------
#  Conversion
#  -----------------------------------------------------------------------------

function convert(::Type{LKJ{T}}, d::LKJ) where T <: Real
    LKJ{T, typeof(d.d)}(d.d, T(d.η), T(d.logc0))
end
Base.convert(::Type{LKJ{T}}, d::LKJ{T}) where {T<:Real} = d

function convert(::Type{LKJ{T}}, d::Integer, η, logc0) where T <: Real
    LKJ{T, typeof(d)}(d, T(η), T(logc0))
end

#  -----------------------------------------------------------------------------
#  Properties
#  -----------------------------------------------------------------------------


size(d::LKJ) = (d.d, d.d)

rank(d::LKJ) = d.d

insupport(d::LKJ, R::AbstractMatrix) = isreal(R) && size(R) == size(d) && isone(Diagonal(R)) && isposdef(R)

mean(d::LKJ) = Matrix{partype(d)}(I, d.d, d.d)

function mode(d::LKJ; check_args::Bool=true)
    @check_args(
        LKJ,
        @setup((_, η) = params(d)),
        (η, η > 1, "mode is defined only when η > 1."),
    )
    return mean(d)
end

function var(lkj::LKJ)
    d = lkj.d
    d > 1 || return zeros(d, d)
    σ² = var(_marginal(lkj))
    σ² * (ones(partype(lkj), d, d) - I)
end

params(d::LKJ) = (d.d, d.η)

@inline partype(d::LKJ{T}) where {T <: Real} = T

#  -----------------------------------------------------------------------------
#  Evaluation
#  -----------------------------------------------------------------------------

function lkj_logc0(d::Integer, η::Real)
    T = float(Base.promote_typeof(d, η))
    d > 1 || return zero(η)
    if isone(η)
        if iseven(d)
            logc0 = -lkj_onion_loginvconst_uniform_even(d, T)
        else
            logc0 = -lkj_onion_loginvconst_uniform_odd(d, T)
        end
    else
        logc0 = -lkj_onion_loginvconst(d, η)
    end
    return logc0
end

logkernel(d::LKJ, R::AbstractMatrix) = (d.η - 1) * logdet(R)

#  -----------------------------------------------------------------------------
#  Sampling
#  -----------------------------------------------------------------------------

function _rand!(rng::AbstractRNG, d::LKJ, R::AbstractMatrix)
    R .= _lkj_onion_sampler(d.d, d.η, rng)
end

function _lkj_onion_sampler(d::Integer, η::Real, rng::AbstractRNG = Random.default_rng())
    #  Section 3.2 in LKJ (2009 JMA)
    #  1. Initialization
    R = ones(typeof(η), d, d)
    d > 1 || return R
    β = η + 0.5d - 1
    u = rand(rng, Beta(β, β))
    R[1, 2] = 2u - 1
    R[2, 1] = R[1, 2]
    #  2.
    for k in 2:d - 1
        #  (a)
        β -= 0.5
        #  (b)
        y = rand(rng, Beta(k / 2, β))
        #  (c)
        u = randn(rng, k)
        u = u / norm(u)
        #  (d)
        w = sqrt(y) * u
        A = cholesky(R[1:k, 1:k]).L
        z = A * w
        #  (e)
        R[1:k, k + 1] = z
        R[k + 1, 1:k] = z'
    end
    #  3.
    return R
end

#  -----------------------------------------------------------------------------
#  The free elements of an LKJ matrix each have the same marginal distribution
#  -----------------------------------------------------------------------------

function _marginal(lkj::LKJ)
    d = lkj.d
    η = lkj.η
    α = η + 0.5d - 1
    AffineDistribution(-1, 2, Beta(α, α))
end

#  -----------------------------------------------------------------------------
#  Test utils
#  -----------------------------------------------------------------------------

function _univariate(d::LKJ)
    check_univariate(d)
    return DiscreteNonParametric([one(d.η)], [one(d.η)])
end

function _rand_params(::Type{LKJ}, elty, n::Int, p::Int)
    n == p || throw(ArgumentError("dims must be equal for LKJ"))
    η = abs(3randn(elty))
    return n, η
end

#  -----------------------------------------------------------------------------
#  Several redundant implementations of the reciprocal integrating constant.
#  If f(R; n) = c₀ |R|ⁿ⁻¹, these give log(1 / c₀).
#  Every integrating constant formula given in LKJ (2009 JMA) is an expression
#  for 1 / c₀, even if they say that it is not.
#  -----------------------------------------------------------------------------

function lkj_onion_loginvconst(d::Integer, η::Real)
    #  Equation (17) in LKJ (2009 JMA)
    T = float(Base.promote_typeof(d, η))
    h = T(1//2)
    α = η + h * d - 1
    loginvconst = (2*η + d - 3)*T(logtwo) + (T(logπ) / 4) * (d * (d - 1) - 2) + logbeta(α, α) - (d - 2) * loggamma(η + h * (d - 1))
    for k in 2:(d - 1)
        loginvconst += loggamma(η + h * (d - 1 - k))
    end
    return loginvconst
end

function lkj_onion_loginvconst_uniform_odd(d::Integer, ::Type{T}) where {T <: Real}
    #  Theorem 5 in LKJ (2009 JMA)
    h = T(1//2)
    loginvconst = (d - 1) * ((d + 1) * (T(logπ) / 4) - (d - 1) * (T(logtwo) / 4) - loggamma(h * (d + 1)))
    for k in 2:2:(d - 1)
        loginvconst += loggamma(T(k))
    end
    return loginvconst
end

function lkj_onion_loginvconst_uniform_even(d::Integer, ::Type{T}) where {T <: Real}
    #  Theorem 5 in LKJ (2009 JMA)
    h = T(1//2)
    loginvconst = d * ((d - 2) * (T(logπ) / 4) + (3 * d - 4) * (T(logtwo) / 4) + loggamma(h * d)) - (d - 1) * loggamma(T(d))
    for k in 2:2:(d - 2)
        loginvconst += loggamma(k)
    end
    return loginvconst
end

function lkj_vine_loginvconst(d::Integer, η::Real)
    #  Equation (16) in LKJ (2009 JMA)
    expsum = zero(η)
    betasum = zero(η)
    for k in 1:d - 1
        α = η + 0.5(d - k - 1)
        expsum += (2η - 2 + d - k) * (d - k)
        betasum += (d - k) * logbeta(α, α)
    end
    loginvconst = expsum * logtwo + betasum
    return loginvconst
end

function lkj_vine_loginvconst_uniform(d::Integer)
    #  Equation after (16) in LKJ (2009 JMA)
    expsum = 0.0
    betasum = 0.0
    for k in 1:d - 1
        α = (k + 1) / 2
        expsum += k ^ 2
        betasum += k * logbeta(α, α)
    end
    loginvconst = expsum * logtwo + betasum
    return loginvconst
end

function lkj_loginvconst_alt(d::Integer, η::Real)
    #  Third line in first proof of Section 3.3 in LKJ (2009 JMA)
    loginvconst = zero(η)
    for k in 1:d - 1
        loginvconst += 0.5k*logπ + loggamma(η + 0.5(d - 1 - k)) - loggamma(η + 0.5(d - 1))
    end
    return loginvconst
end

function corr_logvolume(n::Integer)
    #  https://doi.org/10.4169/amer.math.monthly.123.9.909
    logvol = 0.0
    for k in 1:n - 1
        logvol += 0.5k*logπ + k*loggamma((k+1)/2) - k*loggamma((k+2)/2)
    end
    return logvol
end
