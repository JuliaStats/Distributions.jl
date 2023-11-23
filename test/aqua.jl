using Distributions
using Test

import Aqua

@testset "Aqua" begin
    # Test ambiguities separately without Base and Core
    # Ref: https://github.com/JuliaTesting/Aqua.jl/issues/77
    Aqua.test_all(
        Distributions;
        ambiguities = false,
        # On older Julia versions, installed dependencies are quite old
        # Thus unbound type parameters show up that are fixed in newer versions
        unbound_args = VERSION >= v"1.6",
    )
    # Tests are not reliable on older Julia versions and
    # show ambiguities in loaded packages
    if VERSION >= v"1.6"
        Aqua.test_ambiguities(Distributions)
    end
end
