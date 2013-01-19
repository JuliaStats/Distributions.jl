##
##
## String representations
##
##

function show(io::IO, d::Distribution)
  print(io, @sprintf "%s distribution\n" typeof(d))
  for parameter in typeof(d).names
    if isa(d.(parameter), AbstractArray)
      param = strcat(ucfirst(string(parameter)), ":\n", d.(parameter), "\n")
    else
      param = strcat(ucfirst(string(parameter)), ": ", d.(parameter), "\n")
    end
    print(io, param)
  end
  m = mean(d)
  if isa(m, AbstractArray)
    print(io, strcat("Mean:\n", m, "\n"))
  else
    print(io, strcat("Mean: ", m, "\n"))
  end
  v = var(d)
  if isa(v, AbstractArray)
    print(io, strcat("Variance:\n", v))
  else
    print(io, strcat("Variance: ", v))
  end
end
