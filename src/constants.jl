# Useful math constants for distributions

import Base.@math_const

@math_const twoπ   6.2831853071795864769 big(2.) * π
@math_const √2     1.4142135623730950488 sqrt(big(2.))
@math_const log2π  1.8378770664093454836 log(big(2.)*π)
@math_const √2π    2.5066282746310005024 sqrt(big(2.)*π)
@math_const r√2π   0.3989422804014326779 1/sqrt(big(2.)*π)