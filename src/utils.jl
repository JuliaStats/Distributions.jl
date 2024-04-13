## macro for argument checking

macro strip_linenos(expr)
    return esc(Base.remove_linenums!(expr))
end

"""
    @check_args(
        D,
        @setup(statements...),
        (arg₁, cond₁, message₁),
        (cond₂, message₂),
        ...,
    )

A convenience macro that generates checks of arguments for a distribution of type `D`.

The macro expects that a boolean variable of name `check_args` is defined and generates
the following Julia code:
```julia
Distributions.check_args(check_args) do
    \$(statements...)
    cond₁ || throw(DomainError(arg₁, \$(string(D, ": ", message₁))))
    cond₂ || throw(ArgumentError(\$(string(D, ": ", message₂))))
    ...
end
```

The `@setup` argument can be elided if no setup code is needed. Moreover, error messages
can be omitted. In this case the message `"the condition \$(cond) is not satisfied."` is
used.
"""
macro check_args(D, setup_or_check, checks...)
    # Extract setup statements
    if Meta.isexpr(setup_or_check, :macrocall) && setup_or_check.args[1] == Symbol("@setup")
        setup_stmts = Any[esc(ex) for ex in setup_or_check.args[3:end]]
    else
        setup_stmts = []
        checks = (setup_or_check, checks...)
    end

    # Generate expressions for each condition
    conds_exprs = map(checks) do check
        if Meta.isexpr(check, :tuple, 3)
            # argument, condition, and message specified
            arg = check.args[1]
            cond = check.args[2]
            message = string(D, ": ", check.args[3])
            return :(($(esc(cond))) || throw(DomainError($(esc(arg)), $message)))
        elseif Meta.isexpr(check, :tuple, 2)
            cond_or_message = check.args[2]
            if cond_or_message isa String
                # only condition and message specified
                cond = check.args[1]
                message = string(D, ": ", cond_or_message)
                return :(($(esc(cond))) || throw(ArgumentError($message)))
            else
                # only argument and condition specified
                arg = check.args[1]
                cond = cond_or_message
                message = string(D, ": the condition ", cond, " is not satisfied.")
                return :(($(esc(cond))) || throw(DomainError($(esc(arg)), $message)))
            end
        else
            # only condition specified
            cond = check
            message = string(D, ": the condition ", cond, " is not satisfied.")
            return :(($(esc(cond))) || throw(ArgumentError($message)))
        end
    end

    return @strip_linenos quote
        Distributions.check_args($(esc(:check_args))) do
            $(__source__)
            $(setup_stmts...)
            $(conds_exprs...)
        end
    end
end

"""
    check_args(f, check::Bool)

Perform check of arguments by calling a function `f`.

If `check` is `false`, the checks are skipped.
"""
function check_args(f::F, check::Bool) where {F}
    check && f()
    nothing
end

##### Utility functions

isunitvec(v::AbstractVector) = (norm(v) - 1.0) < 1.0e-12

isprobvec(p::AbstractVector{<:Real}) =
    all(x -> x ≥ zero(x), p) && isapprox(sum(p), one(eltype(p)))

# get a type wide enough to represent all a distributions's parameters
# (if the distribution is parametric)
# if the distribution is not parametric, we need this to be a float so that
# in-place pdf calculations, etc. allocate storage correctly
@inline partype(::Distribution) = Float64

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
