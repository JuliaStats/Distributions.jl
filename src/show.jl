# Display of distributions
#
# `show(io, d)` is the single-line form that containers and string interpolation use. It prints the
# call that constructs `d` from its parameters. Distributions that are constructed by a function,
# such as `truncated`, define it themselves.
#
# `show(io, MIME"text/plain"(), d)` is the multi-line report that the REPL displays. It is built
# from three internal hooks: `_showname` prints the name of `d`, `_showparams` its parameters, and
# `_showdomain` the values that `d` can take. Every line is printed with a leading newline, so that
# a report never ends with one.

show(io::IO, d::Distribution) = _showcall(io, nameof(typeof(d)), namedparams(d)...)

function show(io::IO, ::MIME"text/plain", d::Distribution)
    _showname(io, d)
    print(io, " distribution")
    _showparams(io, d)
    _showdomain(io, d)
    return nothing
end

_showname(io::IO, d::Distribution) = print(io, nameof(typeof(d)))

_showparams(io::IO, d::Distribution) = _showsection(io, "Parameters", namedparams(d))

function _showsection(io::IO, header, nt::NamedTuple)
    isempty(nt) && return nothing
    print(io, '\n', header, ':')
    names = map(string, keys(nt))
    width = maximum(textwidth, names)
    for (name, value) in zip(names, values(nt))
        print(io, "\n  ", rpad(name, width), " = ")
        show(io, value)
    end
    return nothing
end

_showdomain(io::IO, d::Distribution) = nothing
_showdomain(io::IO, d::UnivariateDistribution) = _showsupport(io, d)
_showdomain(io::IO, d::MultivariateDistribution) = print(io, "\nDimension:\n  ", length(d))
_showdomain(io::IO, d::MatrixDistribution) = print(io, "\nSize:\n  ", size(d, 1), '×', size(d, 2))

# The support is described by `minimum` and `maximum` alone: a finite bound belongs to the support,
# as the definition of `in` for `RealInterval` shows.
function _showsupport(io::IO, d::ContinuousUnivariateDistribution)
    lo, hi = extrema(d)
    print(io, "\nSupport:\n  ", lo, isfinite(lo) ? " ≤ x " : " < x ", isfinite(hi) ? "≤ " : "< ", hi)
    return nothing
end

function _showsupport(io::IO, d::DiscreteUnivariateDistribution)
    lo, hi = extrema(d)
    print(io, "\nSupport:\n  {")
    if !isfinite(lo) && !isfinite(hi)
        print(io, "…, -1, 0, 1, …")
    elseif !isfinite(lo)
        print(io, "…, ", hi - 1, ", ", hi)
    elseif !isfinite(hi)
        print(io, lo, ", ", lo + 1, ", …")
    elseif hi - lo < 4
        join(io, lo:hi, ", ")
    else
        print(io, lo, ", ", lo + 1, ", …, ", hi)
    end
    print(io, '}')
    return nothing
end

# Single-line form of a distribution that is constructed by a function, e.g. `truncated`
function _showcall(io::IO, name, args...)
    print(io, name, '(')
    for (i, arg) in enumerate(args)
        i > 1 && print(io, ", ")
        show(io, arg)
    end
    print(io, ')')
    return nothing
end

# Bounds of `Truncated` and `Censored`, one of which is `nothing` if `d` is bounded on one side only
_bounds(lower, upper) = (; lower, upper)
_bounds(lower, ::Nothing) = (; lower)
_bounds(::Nothing, upper) = (; upper)
_bounds(::Nothing, ::Nothing) = (;)
