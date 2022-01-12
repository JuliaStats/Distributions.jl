

# the name of a distribution
#
#   Generally, this should be just the type name, e.g. Normal.
#   Under certain circumstances, one may want to specialize
#   this function to provide a name that is easier to read,
#   especially when the type is parametric.
#
distrname(d::Distribution) = string(typeof(d))

show(io::IO, d::Distribution) = show(io, d, fieldnames(typeof(d)))

# For some distributions, the fields may contain internal details,
# which we don't want to show, this function allows one to
# specify which fields to show.
#
function show(io::IO, d::Distribution, pnames)
    uml, namevals = _use_multline_show(d, pnames)
    uml ? show_multline(io, d, namevals) : show_oneline(io, d, namevals)
end

const _NameVal = Tuple{Symbol,Any}

function _use_multline_show(d::Distribution, pnames)
    # decide whether to use one-line or multi-line format
    #
    # Criteria: if total number of values is greater than 8, or
    # there are params that are neither numbers, tuples, or vectors,
    # we use multi-line format
    #
    namevals = _NameVal[]
    multline = false
    tlen = 0
    for (i, p) in enumerate(pnames)
        pv = getfield(d, p)
        if !(isa(pv, Number) || isa(pv, NTuple) || isa(pv, AbstractVector))
            multline = true
        else
            tlen += length(pv)
        end
        push!(namevals, (p, pv))
    end
    if tlen > 8
        multline = true
    end
    return (multline, namevals)
end

function _use_multline_show(d::Distribution)
    _use_multline_show(d, fieldnames(typeof(d)))
end

function show_oneline(io::IO, d::Distribution, namevals)
    print(io, distrname(d))
    np = length(namevals)
    print(io, '(')
    for (i, nv) in enumerate(namevals)
        (p, pv) = nv
        print(io, p)
        print(io, '=')
        show(io, pv)
        if i < np
            print(io, ", ")
        end
    end
    print(io, ')')
end

function show_multline(io::IO, d::Distribution, namevals; newline=true)
    print(io, distrname(d))
    println(io, "(")
    for (p, pv) in namevals
        print(io, p)
        print(io, ": ")
        println(io, pv)
    end
    newline ? println(io, ")") : print(io, ")")
end
