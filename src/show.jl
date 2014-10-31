

# the name of a distribution
#
#   Generally, this should be just the type name, e.g. Normal.
#   Under certain circumstances, one may want to specialize
#   this function to provide a name that is easier to read, 
#   especially when the type is parametric. 
#
distrname(d::Distribution) = string(typeof(d))

show(io::IO, d::Distribution) = show(io, d, typeof(d).names)

# For some distributions, the fields may contain internal details,
# which we don't want to show, this function allows one to 
# specify which fields to show.
#
function show(io::IO, d::Distribution, pnames::(Symbol...))
    # decide whether to use one-line or multi-line format
    #
    # Criteria: if total number of values is greater than 8, or
    # there are matrix-valued params, we use multi-line format
    #
    namevals = (Symbol, Any)[]
    multline = false
    tlen = 0
    for (i, p) in enumerate(pnames)
        pv = d.(p)
        if !(isa(pv, Number) || isa(pv, (Number...)) || isa(pv, AbstractVector))
            multline = true
        end
        tlen += length(pv)
        push!(namevals, (p, pv))
    end
    if tlen > 8
        multline = true
    end

    # call the function that actually does the job
    multline ? show_multline(io, d, namevals) :
               show_oneline(io, d, namevals)
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

function show_multline(io::IO, d::Distribution, namevals)
    print(io, distrname(d))
    println(io, "(")
    for (p, pv) in namevals
        print(io, p)
        print(io, ": ")
        println(io, pv)
    end
    println(io, ")")
end


