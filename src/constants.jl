# Useful math constants for distributions

import Base.@math_const

@math_const twoπ   6.2831853071795864769 big(2.) * π
@math_const sqrt2     1.4142135623730950488 sqrt(big(2.))
@math_const log2π  1.8378770664093454836 log(big(2.)*π)
@math_const sqrt2π    2.5066282746310005024 sqrt(big(2.)*π)
@math_const logtwo 0.6931471805599453094 log(big(2.))
@math_const loghalf -0.6931471805599453094 log(big(0.5))
