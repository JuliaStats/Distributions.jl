# Contributing

This page details the some of the guidelines that should be followed when contributing to this package.

### Reporting issues & discussing

1. If you need explanation on how to do X, Y using Distributions,
feel free to ask on the Julia [Discourse](https://discourse.julialang.org/c/domain/stats)
or [Slack](https://julialang.slack.com), get an invitation
[here](https://slackinvite.julialang.org/).

2. If you have a bug linked with Distributions, check that it has
not been reported yet on the issues of the repository.
If not, you can file a new issue, add your version of the package
which you can obtain with this command in the Julia REPL:
```julia
julia> ]status Distributions
```

Be exhaustive in your report, give the summary of the bug,
a Minimal Working Example (MWE), what happens and what you
expected to happen.

### Workflow with Git and GitHub

To contribute to the package, fork the repository on GitHub,
clone it and make modifications on a new branch,
**do not commit modifications on master**.
Once your changes are made, push them on your fork and create the
Pull Request on the main repository.

### Requirements

Distributions is a central package which many rely on,
the following are required for contributions to be accepted:
1. Docstrings must be added to all interface and non-trivial functions.
2. Tests validating the modified behavior in the `test` folder. If new test files are added, do not forget to add them in `test/runtests.jl`. Cover possible edge cases. Run the tests locally before submitting the PR.
3. At the end of the tests, `Test.detect_ambiguities(Distributions)` is run to check method ambiguities. Verify that your modified code did not yield method ambiguities.
4. Make according modifications to the `docs` folder, build the documentation locally with `$ julia docs/make.jl`, verify that your modifications display correctly and did not yield warnings.

## Style Guide

Follow the style of the surrounding text when making changes. When adding new features please try to stick to the following points whenever applicable.

### Julia

  * 4-space indentation;
  * modules spanning entire files should not be indented, but modules that have surrounding code should;
  * no blank lines at the start or end of files;
  * do not manually align syntax such as `=` or `::` over adjacent lines;
  * use `function ... end` when a method definition contains more than one toplevel expression;
  * related short-form method definitions don't need a new line between them;
  * unrelated or long-form method definitions must have a blank line separating each one;
  * surround all binary operators with whitespace except for `::`, `^`, and `:`;
  * files containing a single `module ... end` must be named after the module;
  * method arguments should be ordered based on the amount of usage within the method body;
  * methods extended from other modules must follow their inherited argument order, not the above rule;
  * explicit `return` should be preferred except in short-form method definitions;
  * avoid dense expressions where possible e.g. prefer nested `if`s over complex nested `?`s;
  * include a trailing `,` in vectors, tuples, or method calls that span several lines;
  * do not use multiline comments (`#=` and `=#`);
  * wrap long lines as near to 92 characters as possible, this includes docstrings;
  * follow the standard naming conventions used in `Base`.

### Markdown

  * Use unbalanced `#` headers, i.e. no `#` on the right hand side of the header text;
  * include a single blank line between toplevel blocks;
  * unordered lists must use `*` bullets with two preceding spaces;
  * do *not* hard wrap lines;
  * use emphasis (`*`) and bold (`**`) sparingly;
  * always use fenced code blocks instead of indented blocks;
  * follow the conventions outlined in the Julia documentation page on documentation.
