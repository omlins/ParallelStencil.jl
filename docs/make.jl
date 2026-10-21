using Documenter
using DocumenterCitations
using DocumenterVitepress
using Markdown
using ParallelStencil

const EXAMPLEROOT = joinpath(@__DIR__, "..", "examples")

# Return the example script `file` as a Julia code block (used in the `@eval` blocks of the example pages).
mdinclude(file) = Markdown.parse("```julia\n" * read(joinpath(EXAMPLEROOT, file), String) * "\n```")

const bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:numeric)


@info "Building documentation website using DocumenterVitepress.jl..."
makedocs(;
    modules   = [ParallelStencil],
    authors   = "Samuel Omlin, Ludovic Räss",
    sitename  = "ParallelStencil.jl",
    format    = DocumenterVitepress.MarkdownVitepress(
        repo      = "github.com/omlins/ParallelStencil.jl",
        devbranch = "main",
        devurl    = "dev",
    ),
    checkdocs = :exports,
    checkdocs_ignored_modules = [ParallelStencil.ParallelKernel], # Its docstrings are included through the corresponding ParallelStencil docstrings.
    plugins   = [bib],
    pages     = [
        "Home"          => "index.md",
        "Overview"      => "overview.md",
        "Usage"         => "usage.md",
        "Examples"      => [
            "examples/diffusion3D.md",
            "examples/diffusion3D_multixpu.md",
            "examples/more.md",
        ],
        "Miniapps"      => "miniapps.md",
        "Testing"       => "testing.md",
        "API reference" => "api.md",
        "References"    => "references.md",
    ],
)


@info "Deploying docs..."
DocumenterVitepress.deploydocs(;
    repo         = "github.com/omlins/ParallelStencil.jl",
    devbranch    = "main",
    push_preview = true,
)
