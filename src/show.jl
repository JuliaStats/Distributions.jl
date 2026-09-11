# Display of distributions
#
# `show(io, d)` is the single-line form that containers and string interpolation use. It prints how
# `d` is constructed from its parameters. Distributions that are not constructed by their own type,
# such as `truncated`, define it themselves.
#
# `show(io, MIME"text/plain"(), d)` is the multi-line report that the REPL displays. It is built
# from two internal hooks: `_showname` prints the name of `d` and `_showparams` its parameters.
# Every line is printed with a leading newline, so that a report never ends with one.

show(io::IO, d::Distribution) = _showcall(io, nameof(typeof(d)), _namedparams(d)...)

function show(io::IO, ::MIME"text/plain", d::Distribution)
    _showname(io, d)
    print(io, " distribution")
    _showparams(io, d)
    return nothing
end

_showname(io::IO, d::Distribution) = print(io, nameof(typeof(d)))

_showparams(io::IO, d::Distribution) = _showsection(io, "Parameters", _namedparams(d))

# Only the display falls back to the fields, so that `show` works for a distribution that does not
# implement `namedparams`; `params` keeps throwing rather than computing with a guess.
_namedparams(d::Distribution) = applicable(namedparams, d) ? namedparams(d) : _fieldparams(d)

function _fieldparams(d::Distribution)
    T = typeof(d)
    return NamedTuple{fieldnames(T)}(ntuple(i -> getfield(d, i), Val(fieldcount(T))))
end

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
