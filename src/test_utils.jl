module TestUtils

import ..Distributions

function test_mvnormal end

if isdefined(Base, :get_extension) && isdefined(Base.Experimental, :register_error_hint)
    function __init__()
        # Better error message if users forget to load Test
        Base.Experimental.register_error_hint(MethodError) do io, exc, _, _
            if exc.f === test_mvnormal &&
                (Base.get_extension(Distributions, :DistributionsTestExt) === nothing)
                print(io, "\nDid you forget to load Test?")
            end
        end
    end
end

end
