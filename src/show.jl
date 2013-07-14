function show(io::IO, d::Distribution)
    @printf io "%s distribution\n" typeof(d)
    for parameter in typeof(d).names
        if isa(d.(parameter), AbstractArray)
            param = string(ucfirst(string(parameter)),
                           ":\n",
                           d.(parameter),
                           "\n")
        else
            param = string(ucfirst(string(parameter)),
                           ": ",
                           d.(parameter),
                           "\n")
        end
        print(io, param)
    end
end

function compact_show(io::IO, d::Distribution)
    print(io, typeof(d))
    print(io, "( ")
    for parameter in typeof(d).names
        param = string(string(parameter), "=", d.(parameter), " ")
        print(io, param)            
    end        
    print(io, ")")  
end

function show(io::IO, d::UnivariateDistribution)
    compact_show(io, d)
end


