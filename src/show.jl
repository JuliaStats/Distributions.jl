##
##
## String representations
##
##

function show(io::IO, d::Distribution)
  print(io, @sprintf "%s distribution\n" typeof(d))
  for parameter in typeof(d).names
    if isa(d.(parameter), AbstractArray)
      param = string(ucfirst(string(parameter)), ":\n", d.(parameter), "\n")
    else
      param = string(ucfirst(string(parameter)), ": ", d.(parameter), "\n")
    end
    print(io, param)
  end
  m = mean(d)
  if isa(m, AbstractArray)
    print(io, string("Mean:\n", m, "\n"))
  else
    print(io, string("Mean: ", m, "\n"))
  end
  v = var(d)
  if isa(v, AbstractArray)
    print(io, string("Variance:\n", v))
  else
    print(io, string("Variance: ", v))
  end
end
