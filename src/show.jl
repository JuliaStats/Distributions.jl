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
        print(io, string(parameter))
        print(io, "=")
        pv = d.(parameter)
        if isa(pv, AbstractVector)
            print(io, '[')
            if !isempty(pv)
                for i = 1 : length(pv)-1
                    print(io, pv[i])
                    print(io, ", ")
                end
                print(io, pv[end])
            end
            print(io, ']')
        else
            print(io, pv)
        end 
        print(io, " ")         
    end        
    print(io, ")")  
end

function show(io::IO, d::UnivariateDistribution)
    compact_show(io, d)
end


