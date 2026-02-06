# NOTE: This file contains many parts that are copied from the file runtests.jl from the Package MPI.jl.
push!(LOAD_PATH, "../src")

import ParallelStencil # Precompile it.
import ParallelStencil: SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_KERNELABSTRACTIONS
@static if (PKG_CUDA in SUPPORTED_PACKAGES) import CUDA end
@static if (PKG_AMDGPU in SUPPORTED_PACKAGES) import AMDGPU end
@static if (PKG_METAL in SUPPORTED_PACKAGES && Sys.isapple()) import Metal end
@static if (PKG_KERNELABSTRACTIONS in SUPPORTED_PACKAGES) import KernelAbstractions end # KernelAbstractions does not require extra harness env vars beyond the existing CUDA/AMDGPU settings.

excludedfiles = [ "test_excluded.jl", "test_incremental_compilation.jl", "test_revise.jl"]; # TODO: test_incremental_compilation has to be deactivated until Polyester support released

function runtests(testfiles=String[])
    exename   = joinpath(Sys.BINDIR, Base.julia_exename())
    testdir   = pwd()
    istest(f) = endswith(f, ".jl") && startswith(basename(f), "test_")
    testfiles = isempty(testfiles) ? sort(filter(istest, vcat([joinpath.(root, files) for (root, dirs, files) in walkdir(testdir)]...))) : testfiles

    nabort = 0
    nfail  = 0
    abortfiles = String[]
    failfiles  = String[]
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

    if (PKG_KERNELABSTRACTIONS in SUPPORTED_PACKAGES && !KernelAbstractions.functional(KernelAbstractions.CPU()))
        @warn "Test Skip: All KernelAbstractions tests will be skipped because KernelAbstractions is not functional (if this is unexpected type `import KernelAbstractions; KernelAbstractions.functional(KernelAbstractions.CPU())` to debug your KernelAbstractions installation)."
    end

    for f in testfiles
        println("")
        if basename(f) ∈ excludedfiles
            println("Test Skip:")
            println("$f")
            continue
        end
        cmd = `$exename --color=yes -O3 --startup-file=no $(joinpath(testdir, f))`
        stdout_path = tempname()
        stderr_path = tempname()
        stdout_content = ""
        stderr_content = ""
        proc = nothing
        try
            open(stdout_path, "w") do stdout_io
                open(stderr_path, "w") do stderr_io
                    proc = run(pipeline(Cmd(cmd; ignorestatus=true), stdout=stdout_io, stderr=stderr_io); wait=true)
                end
            end
            stdout_content = read(stdout_path, String)
            stderr_content = read(stderr_path, String)
            print(stdout_content)
            print(Base.stderr, stderr_content)
        catch ex
            println("Test Abort: a system-level exception occurred while running the test file $f :")
            println(ex)
            nabort += 1
            push!(abortfiles, f)
            continue
        finally
            if ispath(stdout_path)
                rm(stdout_path; force=true)
            end
            if ispath(stderr_path)
                rm(stderr_path; force=true)
            end
        end
        if !occursin(r"(?i)test summary", stdout_content)
            nabort += 1
            push!(abortfiles, f)
        elseif proc !== nothing && !success(proc)
            nfail += 1
            push!(failfiles, f)
        end
    end
    println("")
    if nabort == 0 && nfail == 0
        printstyled("Test suite: all selected test files executed and all tests passed.\n"; bold=true, color=:green)
    else
        if nfail > 0
            printstyled("Test suite: $nfail test files(s) have tests that failed or errored (see above for results); files with failed/errored tests:\n"; bold=true, color=:red)
            for f in failfiles
                println(" - $f")
            end
        end
        if nabort > 0
            printstyled("Test suite: $nabort test file(s) aborted execution due to fatal error (see above for details); files aborting execution:\n"; bold=true, color=:red)
            for f in abortfiles
                println(" - $f")
            end
        else
            printstyled("Test suite: all selected test files executed, but some test file(s) have tests that failed or errored (see message above).\n"; bold=true, color=:red)
        end
    end
    println("")
    return nabort+nfail
end

exit(runtests(ARGS))
