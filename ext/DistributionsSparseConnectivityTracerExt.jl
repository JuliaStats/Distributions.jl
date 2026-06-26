module DistributionsSparseConnectivityTracerExt

using Distributions: Distributions
using SparseConnectivityTracer: AbstractTracer

# Only the global `TracerSparsityDetector` is problematic for `@check_args`: it uses primal-free
# tracers (`GradientTracer`/`HessianTracer`, the subtypes of `AbstractTracer`), so the validity
# comparisons (e.g. `σ ≥ 0`) return a tracer that cannot be used in boolean context. Marking these
# as not checkable skips validation so distribution constructors can still be traced.
#
# Local sparsity detection (`TracerLocalSparsityDetector`) passes `Dual` values, which are `<: Real`
# (not `<: AbstractTracer`) and carry a primal, so the comparisons evaluate to `Bool` as usual — the
# default `_arg_checkable` applies there and the checks run normally.
Distributions._arg_checkable(::AbstractTracer) = false

end
