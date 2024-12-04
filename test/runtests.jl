# NOTE: This file contains many parts that are copied from the file runtests.jl from the Package MPI.jl.
push!(LOAD_PATH, "../src")

import ParallelStencil # Precompile it.
import ParallelStencil: SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL
@static if (PKG_CUDA in SUPPORTED_PACKAGES) import CUDA end
@static if (PKG_AMDGPU in SUPPORTED_PACKAGES) import AMDGPU end
@static if (PKG_METAL in SUPPORTED_PACKAGES && Sys.isapple()) import Metal end

excludedfiles = [ "test_excluded.jl", "test_incremental_compilation.jl"]; # TODO: test_incremental_compilation has to be deactivated until Polyester support released

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

    if (PKG_AMDGPU in SUPPORTED_PACKAGES && !AMDGPU.functional())
        @warn "Test Skip: All AMDGPU tests will be skipped because AMDGPU is not functional (if this is unexpected type `import AMDGPU; AMDGPU.functional()` to debug your AMDGPU installation)."
    end

    if (PKG_METAL in SUPPORTED_PACKAGES && (!Sys.isapple() || !Metal.functional()))
        @warn "Test Skip: All Metal tests will be skipped because Metal is not functional (if this is unexpected type `import Metal; Metal.functional()` to debug your Metal installation)."
    end

    for f in testfiles
        println("")
        if basename(f) ∈ excludedfiles
            println("Test Skip:")
            println("$f")
            continue
        end
        try
            run(`$exename -O3 --startup-file=no $(joinpath(testdir, f))`)
        catch ex
            nfail += 1
        end
    end
    return nfail
end

exit(runtests())
