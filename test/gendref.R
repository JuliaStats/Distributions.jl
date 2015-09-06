# A R script to generate reference values for testing distributions
#
#   This script will take the following two list of distribution
#   entries as input:
#
#       - continuous_test.lst
#       - discrete_test.lst
#
#   The generated results will be written to the following files,
#   in JSON format, respectively
#
#       - continuous_test.json
#       - discrete_test.json
#
#   To run this script:
#
#       Rscript gendref.R
#
#   Note: this scripts depends on a number of R libraries for
#         file I/O and the computation of statistical functions.
#

read.entries <- function(filename) {
    # Read a list of entries from a given file
    lst <- read.table(filename, header=FALSE,
        sep="\t", blank.lines.skip = TRUE, comment.char="#")
    as.vector(lst$V1)
}

parse.entry <- function(entry) {
    # Parse an entry into a distribution name, and
    # a vector of argument values

    terms <- unlist(strsplit(entry, "\\s*,\\s*|\\(|\\)"))
    nt <- length(terms)
    list(name=terms[1],
         args=as.numeric(tail(terms, nt-1)))
}


distr.info <- function(entry, args) {
    # Extract the distribution information given an entry,
    # and return a list with following fields:
    #
    #   - name:    The distribution name
    #   - args:    Full argument list
    #   - supp:    A vector to represent the support as c(min, max)
    #   - logpdf:  The function to compute logpdf
    #   - cdf:     The function to compute cdf
    #   - props:   A list of named properties to be verified
    #

    parsed <- parse.entry(entry)
    dname <- parsed$name
    dargs <- parsed$args

    switch (dname,
        Bernoulli        = bernoulli.info(dargs),
        Binomial         = binomial.info(dargs),
        DiscreteUniform  = discrete.uniform.info(dargs),
        Geometric        = geometric.info(dargs),
        Hypergeometric   = hypergeometric.info(dargs),
        NegativeBinomial = negative.binomial.info(dargs),
        Poisson          = poisson.info(dargs)
    )
}


########################################
#
#  Distribution information
#
########################################

get.arg <- function(a, i, v0) {
    if (!is.na(a) && i <= length(a)) { a[i] } else { v0 }
}

xlogx <- function(x) {
    if (x == 0) { 0.0 } else { x * log(x) }
}

bernoulli.info <- function(args) {
    stopifnot(length(args) <= 1)
    p <- get.arg(args, 1, 0.5)
    q <- 1.0 - p
    list(name="Bernoulli",
         args=c(p),
         supp=c(0, 1),
         logpdf=function(x){ dbinom(x, 1, p, log=TRUE) },
         cdf=function(x) { pbinom(x, 1, p) },
         quan=function(v) { qbinom(v, 1, p) },
         props=list(
             succprob=p,
             failprob=q,
             mean=p,
             median= if(p<0.5) {0} else if (p > 0.5) {1} else {0.5},
             var=p * q,
             skewness=(1 - 2*p) / sqrt(p*q),
             kurtosis=(1 - 6*p*q) / (p*q),
             entropy=-(xlogx(p) + xlogx(q))))
}

binomial.info <- function(args) {
    stopifnot(length(args) <= 2)
    n <- get.arg(args, 1, 1)
    p <- get.arg(args, 2, 0.5)
    q <- 1.0 - p
    list(name="Binomial",
         args=c(n, p),
         supp=c(0, n),
         logpdf=function(x){ dbinom(x, n, p, log=TRUE)},
         cdf=function(x) { pbinom(x, n, p) },
         quan=function(v) { qbinom(v, n, p) },
         props=list(
             succprob=p,
             failprob=q,
             ntrials=n,
             mean=n * p,
             median=floor(n * p),
             var=n * p * q,
             skewness=(q - p) / sqrt(n*p*q),
             kurtosis=(1 - 6*p*q) / (n*p*q)))
}

discrete.uniform.info <- function(args) {
    stopifnot(length(args) <= 2)
    nargs <- length(args)
    a <- 0
    b <- 1
    if (nargs == 1) {
        b <- args[1]
    } else if (nargs == 2) {
        a <- args[1]
        b <- args[2]
    }
    s<- b - a + 1
    list(name="DiscreteUniform",
         args=c(a, b),
         supp=c(a, b),
         props=list(
             span=s,
             probval=1/s,
             mean=(a + b)/2,
             median=(a + b)/2,
             var=(s^2 - 1)/12,
             skewness=0,
             kurtosis=-(6 * (s^2 + 1))/(5 * (s^2 - 1)),
             entropy=log(s)))
}

geometric.info <- function(args) {
    stopifnot(length(args) <= 1)
    p <- get.arg(args, 1, 0.5)
    list(name="Geometric",
         args=c(p),
         supp=c(0, Inf),
         logpdf=function(x){ dgeom(x, p, log=TRUE)},
         cdf=function(x) { pgeom(x, p) },
         quan=function(v) { qgeom(v, p) },
         props=list(
             succprob=p,
             failprob=1.0-p,
             mean=(1-p)/p,
             var=(1-p)/p^2,
             skewness=(2-p)/sqrt(1-p),
             kurtosis=6 + p^2/(1-p),
             entropy=-((1-p)*log2(1-p) + p * log2(p))/p))
}

hypergeometric.info <- function(args) {
    stopifnot(length(args) == 3)
    ns <- args[1]
    nf <- args[2]
    n <- args[3]

    # Wikipedia's notation
    N <- ns + nf
    K <- ns
    mean.val <- n * (K / N)
    var.val  <- n * (K / N) * ((N-K) / N) * ((N - n) / (N - 1))
    skew.val <- ((N - 2*K) * sqrt(N - 1) * (N - 2*n)) /
                (sqrt(n * K * (N - K) * (N - n)) * (N - 2))

    kurt.num <- (N-1) * N^2 * (N * (N+1)- 6 * K * (N-K) - 6 * n * (N-n)) +
                6 * n * K * (N-K) * (N-n) * (5*N-6)
    kurt.den <- n * K * (N - K) * (N - n) * (N - 2) * (N - 3)
    kurt.val <- kurt.num / kurt.den

    list(name="Hypergeometric",
         args=c(ns, nf, n),
         supp=c(max(0, n+K-N), min(n, K)),
         logpdf=function(x){ dhyper(x, ns, nf, n, log=TRUE)},
         cdf=function(x) { phyper(x, ns, nf, n) },
         quan=function(v) { qhyper(v, ns, nf, n) },
         props=list(
             mean=mean.val,
             var=var.val,
             skewness=skew.val,
             kurtosis=kurt.val))
}

negative.binomial.info <- function(args) {
    stopifnot(length(args) <= 2)
    r <- get.arg(args, 1, 1.0)
    p <- get.arg(args, 2, 0.5)
    list(name="NegativeBinomial",
         args=c(r, p),
         supp=c(0, Inf),
         logpdf=function(x){ dnbinom(x, size=r, prob=p, log=TRUE)},
         cdf=function(x) { pnbinom(x, size=r, prob=p) },
         quan=function(v) { qnbinom(v, size=r, prob=p) },
         props=list(
             succprob=p,
             failprob=1.0-p,
             mean=p * r / (1-p),
             var=p * r / (1-p)^2,
             skewness=(1 + p) / sqrt(p * r),
             kurtosis=6 / r + (1-p)^2 / (p*r)))
}

poisson.info <- function(args) {
    stopifnot(length(args) <= 1)
    lam <- get.arg(args, 1, 1.0)
    u <- if (lam > 0) { Inf } else { 0 }
    list(name="Poisson",
         args=c(lam),
         supp=c(0, u),
         logpdf=function(x){ dpois(x, lam, log=TRUE)},
         cdf=function(x) { ppois(x, lam) },
         quan=function(v) { qpois(v, lam) },
         props=list(
             rate=lam,
             mean=lam,
             var=lam,
             skewness=1/sqrt(lam),
             kurtosis=1/lam))
}


########################################
#
#  Point wise evaluation
#
########################################

eval.samples <- function(qfun, dmin, dmax, is.discrete=TRUE) {
    if (is.discrete) {
        vmin <- if (is.finite(dmin)) { dmin } else { qfun(0.01) }
        vmax <- if (is.finite(dmax)) { dmax } else { qfun(0.99) }

        if (vmax - vmin + 1 <= 10) {
            seq(vmin, vmax)
        } else {
            xs <- unique(round(qfun(seq(0.1, 0.9, 0.1))))
            if (vmin < xs[1]) {
                xs <- c(vmin, xs)
            }
            if (vmax > xs[length(xs)]) {
                xs <- c(xs, vmax)
            }
            xs
        }
    } else {
        qfun(seq(0.1, 0.9, 0.1))
    }
}

eval.points <- function(info, is.discrete=TRUE) {
    pts <- list()
    lpdf <- info$logpdf
    cdf  <- info$cdf
    quan <- info$quan

    if (!is.null(lpdf) && !is.null(cdf) && !is.null(quan)) {
        dmin <- info$supp[1]
        dmax <- info$supp[2]
        xs <- eval.samples(quan, dmin, dmax, is.discrete)
        pts$x = xs
        pts$lp = lpdf(xs)
        pts$cdf = cdf(xs)
    }
    pts
}


########################################
#
#  JSON output
#
########################################

json.str <- function(val) {
    # convert a value to json string
    v <- val
    if (is.numeric(val) && !is.finite(val)) {
        if (val > 0) {
            v <- "inf"
        } else if (val < 0) {
            v <- "-inf"
        } else {
            v <- "nan"
        }
    }

    if (is.numeric(v)) {
        if (v == floor(v)) {
            sprintf("%g", v)
        } else {
            sprintf("%.16g", v)
        }
    } else {
        sprintf("\"%s\"", v)
    }
}

cat.attribute <- function (prefix, name, val, last=FALSE) {
    s <- sprintf("\"%s\": %s", name, json.str(val))
    if (last) {
        cat(prefix, s, "\n", sep="")
    } else {
        cat(prefix, s, ",\n", sep="")
    }
}

cat.point <- function (prefix, points, i) {
    n <- length(points$x)
    x <- points$x[i]
    lp <- points$lp[i]
    cd <- points$cdf[i]

    s <- sprintf("{ \"x\": %s, \"logpdf\": %s, \"cdf\": %s }",
        x, lp, cd)

    if (i == n) {
        cat(prefix, s, "\n", sep="")
    } else {
        cat(prefix, s, ",\n", sep="")
    }
}


write.json <- function(expr, info, points, last=FALSE) {
    cat("{\n")
    cat.attribute("  ", "expr", expr)
    cat.attribute("  ", "dtype", info$name)
    cat.attribute("  ", "minimum", info$supp[1])
    cat.attribute("  ", "maximum", info$supp[2])

    # output properties
    cat("  \"properties\": {\n")
    props <- info$props
    i <- 0
    n <- length(props)
    for (pn in names(props)) {
        i <- i + 1
        cat.attribute("    ", pn, props[[pn]], last=(i==n))
    }
    cat("  },\n")

    # output points
    n <- length(points$x)
    if (n > 0) {
        cat("  \"points\": [\n")
        for (i in 1:n) {
            cat.point("    ", points, i)
        }
        cat("  ]\n")
    } else {
        cat("  \"points\": []\n")
    }

    # closing
    if (last) {
        cat("}\n")
    } else {
        cat("},\n")
    }
}


########################################
#
#  Main program
#
########################################

do.main <- function(lstname) {
    # read a list of entries from file <lstname>.lst
    cat("For list", lstname, "\n")
    cat("-------------------------------\n")
    lstfile <- sprintf("%s.lst", lstname)
    entries <- read.entries(lstfile)

    # derive information
    infolist <- list()
    ptslist <- list()
    n <- 0
    for (e in entries) {
        info <- distr.info(e)
        cat("On", e, "-->", info$name, ":",
            paste(info$args, collapse=", "), "\n")
        n <- n + 1
        infolist[[n]] = info
        ptslist[[n]] = eval.points(info)
    }

    # output
    outfile = sprintf("%s.ref.json", lstname)
    sink(outfile)
    cat("[\n")
    for (i in 1:n) {
        write.json(entries[i], infolist[[i]], ptslist[[i]], last=(i==n))
    }
    cat("]\n")
    sink()

    cat("\n")
}

do.main("discrete_test")
