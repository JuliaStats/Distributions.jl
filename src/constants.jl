# Useful math constants for distributions

import Base.@math_const

@math_const halfπ  1.5707963267948966192 big(π) * 0.5
@math_const twoπ   6.2831853071795864769 big(2.) * π
@math_const sqrt2  1.4142135623730950488 sqrt(big(2.))
@math_const sqrt3  1.7320508075688772935 sqrt(big(3.))
@math_const invπ   0.3183098861837906715 big(1.) / π
@math_const logπ   1.1447298858494001741 log(big(π))
@math_const log2π  1.8378770664093454836 log(big(2.)*π)
@math_const log4π  2.5310242469692907930 log(big(4.)*π)
@math_const sqrt2π 2.5066282746310005024 sqrt(big(2.)*π)
@math_const logtwo 0.6931471805599453094 log(big(2.))
@math_const loghalf -0.6931471805599453094 log(big(0.5))

@math_const sqrthalfπ 1.2533141373155002512 sqrt(big(0.5)*π)
@math_const sqrt2onπ 0.7978845608028653559 sqrt(big(2.)/π)
@math_const logsqrt2onπ -0.225791352644727432 log(sqrt(big(2.)/π))
