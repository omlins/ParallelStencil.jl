# NOTE: This file contains many parts that are copied from the file runtests.jl from the Package MPI.jl.
push!(LOAD_PATH, "../src")

import ParallelStencil # Precompile it.
import ParallelStencil: SUPPORTED_PACKAGES, PKG_CUDA
@static if (PKG_CUDA in SUPPORTED_PACKAGES) import CUDA end

excludedfiles = [ "test_excluded.jl"];

function runtests()
    exename   = joinpath(Sys.BINDIR, Base.julia_exename())
    testdir   = pwd()
    istest(f) = endswith(f, ".jl") && startswith(basename(f), "test_")
    testfiles = sort(filter(istest, vcat([joinpath.(root, files) for (root, dirs, files) in walkdir(testdir)]...)))

    nfail = 0
    printstyled("Testing package ParallelStencil.jl\n"; bold=true, color=:white)

    if (PKG_CUDA in SUPPORTED_PACKAGES && !CUDA.functional())
        @warn "Test Skip: All CUDA tests will be skipped because CUDA is not functional (if this is unexpected type `import CUDA; CUDA.functional(true)` to debug your CUDA installation)."
    end

    for f in testfiles
        println("")
        if f ∈ excludedfiles
            println("Test Skip:")
            println("$f")
            continue
        end
        try
            run(`$exename -O3 --check-bounds=no $(joinpath(testdir, f))`)
        catch ex
            nfail += 1
        end
    end
    return nfail
end

exit(runtests())
