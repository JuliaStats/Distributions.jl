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

source("rdistributions.R")

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


get.distr <- function(entry) {
    # Get a distribution object based on a given entry
    parsed <- parse.entry(entry)
    dname <- parsed$name
    dargs <- parsed$args

    switch (dname,
        Bernoulli        = get.bernoulli(dargs),
        Binomial         = get.binomial(dargs),
        DiscreteUniform  = get.discrete.uniform(dargs),
        Geometric        = get.geometric(dargs),
        Hypergeometric   = get.hypergeometric(dargs),
        NegativeBinomial = get.negative.binomial(dargs),
        Poisson          = get.poisson(dargs)
    )
}


########################################
#
#  Get Distributions
#
########################################

get.arg <- function(a, i, v0) {
    if (!is.na(a) && i <= length(a)) { a[i] } else { v0 }
}

get.bernoulli <- function(args) {
    stopifnot(length(args) <= 1)
    new("Bernoulli", p=get.arg(args, 1, 0.5))
}

get.binomial <- function(args) {
    stopifnot(length(args) <= 2)
    new("Binomial",
        n=get.arg(args, 1, 1),
        p=get.arg(args, 2, 0.5))
}

get.discrete.uniform <- function(args) {
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
    new("DiscreteUniform", a=a, b=b)
}

get.geometric <- function(args) {
    stopifnot(length(args) <= 1)
    new("Geometric", p=get.arg(args, 1, 0.5))
}

get.hypergeometric <- function(args) {
    stopifnot(length(args) == 3)
    new("Hypergeometric", ns=args[1], nf=args[2], n=args[3])
}

get.negative.binomial <- function(args) {
    stopifnot(length(args) <= 2)
    new("NegativeBinomial",
        r=get.arg(args, 1, 1.0),
        p=get.arg(args, 2, 0.5))
}

get.poisson <- function(args) {
    stopifnot(length(args) <= 1)
    new("Poisson", lambda=get.arg(args, 1, 1.0))
}


########################################
#
#  Point wise evaluation
#
########################################

eval.samples <- function(distr) {
    if (is.discrete(distr)) {
        su <- supp(distr)
        dmin <- su[1]
        dmax <- su[2]
        vmin <- if (is.finite(dmin)) { dmin } else { quan(distr, 0.01) }
        vmax <- if (is.finite(dmax)) { dmax } else { quan(distr, 0.99) }

        if (vmax - vmin + 1 <= 10) {
            seq(vmin, vmax)
        } else {
            xs <- unique(round(quan(distr, seq(0.1, 0.9, 0.1))))
            if (vmin < xs[1]) {
                xs <- c(vmin, xs)
            }
            if (vmax > xs[length(xs)]) {
                xs <- c(xs, vmax)
            }
            xs
        }
    } else {
        quan(distr, seq(0.1, 0.9, 0.1))
    }
}

eval.points <- function(distr) {
    xs <- eval.samples(distr)
    list(x=xs,
         pd=pd(distr, xs),
         lp=logpd(distr, xs),
         cp=cd(distr, xs))
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
    p <- points$pd[i]
    lp <- points$lp[i]
    cd <- points$cp[i]

    s <- sprintf("{ \"x\": %s, \"pdf\": %s, \"logpdf\": %s, \"cdf\": %s }",
        x, p, lp, cd)

    if (i == n) {
        cat(prefix, s, "\n", sep="")
    } else {
        cat(prefix, s, ",\n", sep="")
    }
}


write.json <- function(expr, distr, points, last=FALSE) {
    cat("{\n")
    dname <- class(distr)[1]
    suppr <- supp(distr)
    cat.attribute("  ", "expr", expr)
    cat.attribute("  ", "dtype", dname)
    cat.attribute("  ", "minimum", suppr[1])
    cat.attribute("  ", "maximum", suppr[2])

    # output properties
    cat("  \"properties\": {\n")
    props <- properties(distr)
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
    distrlist <- list()
    ptslist <- list()
    n <- 0
    for (e in entries) {
        distr <- get.distr(e)
        dname <- class(distr)[1]
        cat("On ", e, " --> ", dname, ": ", sep="")
        for (sn in slotNames(distr)) {
            cat(sn, "=", slot(distr, sn), " ", sep="")
        }
        cat("\n")
        n <- n + 1
        distrlist[[n]] = distr
        ptslist[[n]] = eval.points(distr)
    }

    # output
    outfile = sprintf("%s.ref.json", lstname)
    sink(outfile)
    cat("[\n")
    for (i in 1:n) {
        write.json(entries[i], distrlist[[i]], ptslist[[i]], last=(i==n))
    }
    cat("]\n")
    sink()

    cat("\n")
}

do.main("discrete_test")
