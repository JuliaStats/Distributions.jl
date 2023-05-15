# R References for Distributions.jl

We rely on the distribution-related functions provided by
[R](https://www.r-project.org) and a number of packages in
[CRAN](https://cran.r-project.org) to generate references
for verifying the correctness of our implementations.

## Dependencies

The reference implementation depends on several R packages
in addition to the R language itself:

| R packages  |  Functionalities |
| ----------- | ---------------- |
| stringr     | For string parsing  |
| R6          | OOP for implementing distributions |
| extraDistr  | A number of distributions |
| VGAM        | For ``Frechet`` and ``Levy`` |
| distr       | For ``Arcsine`` |
| chi         | For ``Chi`` |
| circular    | For ``VonMises`` |
| statmod     | For ``InverseGaussian`` |
| skellam     | For ``Skellam`` |
| BiasedUrn   | For ``NoncentralHypergeometric`` |
| fBasics     | For ``NormalInverseGaussian`` |
| gnorm       | For ``PGeneralizedGaussian`` |
| LindleyR    | For ``Lindley`` |
| ExtDist     | For ``JohnsonSU`` |

## Usage

All reference classes are in ``test/ref/continuous`` and ``test/ref/discrete``. One R file for each distribution class.

The test entries are listed in ``test/ref/continuous_test.lst`` and ``test/ref/discrete_test.lst``. Each entry is a Julia statement for constructing a distribution. The entries can be commented out using ``#``.

One can enter ``Rscript gendref.R`` **within the directory** ``test/ref`` to generate the reference data files: ``test/ref/continuous_test.ref.json`` and ``test/ref/discrete_test.ref.json``.

The testing script ``test/univariate.jl`` loads these reference data files to verify the implementations of Julia distribution classes.

**Note:** The reference data files are under version control. You may only regenerate them only when you are implementing new distribution classes or adding testing cases.

## Reference Classes

Each reference distribution class is implemented as an
[R6 class](https://cran.r-project.org/web/packages/R6/vignettes/Introduction.html),
and provides a uniform interface described as follows:

Let ``D`` be the reference class, and ``d`` be an reference object:

- Inherit from either ``DiscreteDistribution`` or ``ContinuousDistribution``.
- ``d <- D$new(...)``
    - construct a distribution object in a way similar to the counterpart in Julia.
    - Within the ``R6`` class definition, this is implemented via the ``initialize`` method.
- ``d$supp()``
    - return a vector in the form of ``c(l, r)``, where ``l`` and ``r`` are respectively the left and right bounds of the support range.
- ``d$properties()``
    - return a list of named properties, including parameters and stats, *e.g.* ``scale``, ``mean``, etc.
    - Let ``dj`` be a corresponding Julia distribution instance and ``props <- d$properties()``. Then ``props$xxx`` should be *(nearly)* equal to ``xxx(jd)``.
- ``d$pdf(x, log=FALSE)``
    - return the pdf values at a vector ``x``.
    - When ``log`` is set to ``TRUE``, it returns log-pdf values.
- ``d$cdf(x)``
    - return the cumulative probability values at ``x``.
- ``d$quan(v)``
    - return the quantiles at values in ``v``.

### Example

Here is an example for ``Normal`` distribution.

```r
Normal <- R6Class("Normal",
    inherit = ContinuousDistribution,
    public = list(
        names = c("mu", "sigma"),
        mu = NA,
        sigma = NA,
        initialize = function(u=0, s=1) {
            self$mu <- u
            self$sigma <- s
        },
        supp = function() { c(-Inf, Inf) },
        properties = function() {
            u <- self$mu
            s <- self$sigma
            list(location=u,
                 scale=s,
                 mean=u,
                 var=s^2,
                 entropy=(log(2 * pi) + 1 + log(s^2))/2,
                 skewness=0,
                 kurtosis=0)
        },
        pdf = function(x, log=FALSE) {
            dnorm(x, self$mu, self$sigma, log=log)
        },
        cdf = function(x) {
            pnorm(x, self$mu, self$sigma)
        },
        quan = function(v) {
            qnorm(v, self$mu, self$sigma )
        }
    )
)
```

When the testing script sees an entry ``Normal(3, 2)``, it will
call the R expression ``Normal$new(3, 2)`` to construct a reference distribution
for verification.

### Mock Constructors

For certain distributions, Julia provides alternative ways to construct them,
using different parameterization, *e.g.* ``NormalCanon``.
For such cases, one can write a *mock* constructor -- a list with a ``new`` function
to emulate the behavior:

```r
NormalCanon = list(
    new = function(c1=0, c2=1) {
        Normal$new(c1/c2, 1/sqrt(c2))
    }
)
```
