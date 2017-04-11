
TriangularDist = R6Class("TriangularDist",
    inherit = ContinuousDistribution,
    public =list(
        names = c("a", "b", "c"),
        a = NA,
        b = NA,
        c = NA,
        initialize = function(a, b, c=(a+b)/2) {
            self$a <- a
            self$b <- b
            self$c <- c
        },
        supp = function() { c(self$a, self$b) },
        properties = function() {
            a <- self$a
            b <- self$b
            c <- self$c
            m <- (a + b) / 2
            list(mean = (a + b + c) / 3,
                 mode = c,
                 median = if (c >= m) {
                     a + sqrt((b - a) * (c - a) / 2)
                 } else {
                     b - sqrt((b - a) * (b - c) / 2)
                 },
                 var = (a^2 + b^2 + c^2 - a*b - a*c - b*c) / 18,
                 skewness = (sqrt(2)/5) * (a + b - 2*c) * (2*a - b - c) * (a - 2*b + c) /
                    (a^2 + b^2 + c^2 - a*b - a*c - b*c)^1.5,
                 kurtosis = -0.6,
                 entropy = 0.5 + log((b - a) / 2))
        },
        pdf = function(x, log=FALSE) { dtriang(x, self$a, self$b, self$c, log=log) },
        cdf = function(x) { ptriang(x, self$a, self$b, self$c) },
        quan = function(v) { qtriang(v, self$a, self$b, self$c) }
    )
)

SymTriangularDist = list(
    new = function(u=0, s=1) {
        TriangularDist$new(u-s, u+s, u)
    }
)
