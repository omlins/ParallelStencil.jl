# NOTE: This file contains many parts that are copied from the file runtests.jl from the Package MPI.jl.
push!(LOAD_PATH, "../src")

import ParallelStencil # Precompile it.
import ParallelStencil: SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL
@static if (PKG_CUDA in SUPPORTED_PACKAGES) import CUDA end
@static if (PKG_AMDGPU in SUPPORTED_PACKAGES) import AMDGPU end
@static if (PKG_METAL in SUPPORTED_PACKAGES && Sys.isapple()) import Metal end

excludedfiles = [ "test_excluded.jl", "test_incremental_compilation.jl", "test_revise.jl"]; # TODO: test_incremental_compilation has to be deactivated until Polyester support released

function runtests(testfiles=String[])
    exename   = joinpath(Sys.BINDIR, Base.julia_exename())
    testdir   = pwd()
    istest(f) = endswith(f, ".jl") && startswith(basename(f), "test_")
    testfiles = isempty(testfiles) ? sort(filter(istest, vcat([joinpath.(root, files) for (root, dirs, files) in walkdir(testdir)]...))) : testfiles

    nerror = 0
    errorfiles = String[]
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
        cmd = `$exename -O3 --startup-file=no $(joinpath(testdir, f))`
        stdout_path = tempname()
        stderr_path = tempname()
        stdout_content = ""
        stderr_content = ""
        try
            open(stdout_path, "w") do stdout_io
                open(stderr_path, "w") do stderr_io
                    try
                        proc = run(pipeline(Cmd(cmd), stdout=stdout_io, stderr=stderr_io); wait=false)
                        wait(proc) 
                    catch ex
                        println("Test Error: an exception occurred while running the test file $f :")
                        println(ex)
                        nerror += 1
                        push!(errorfiles, f)
                    end
                end
            end
            stdout_content = read(stdout_path, String)
            stderr_content = read(stderr_path, String)
            print(stdout_content)
            print(Base.stderr, stderr_content)
        catch ex
            println("Test Error: an exception occurred while running the test file $f :")
            println(ex)
        finally
            if ispath(stdout_path)
                rm(stdout_path; force=true)
            end
            if ispath(stderr_path)
                rm(stderr_path; force=true)
            end
        end
        if !occursin(r"(?i)test summary", stdout_content)
            nerror += 1
            push!(errorfiles, f)
        end
    end
    println("")
    if nerror == 0
        printstyled("Test suite: all selected test files executed (see above for results).\n"; bold=true, color=:green)
    else
        printstyled("Test suite: $nerror test file(s) aborted execution due to error (see above for details); files aborting execution:\n"; bold=true, color=:red)
        for f in errorfiles
            println(" - $f")
        end
    end
    println("")
    return nerror
end

exit(runtests(ARGS))
