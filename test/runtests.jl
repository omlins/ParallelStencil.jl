# NOTE: This file contains many parts that are copied from the file runtests.jl from the Package MPI.jl.
push!(LOAD_PATH, "../src")

const PREIMPORT_STDERR_SUPPRESSION_RULES = (
    (name="Metal OS support warnings", start=r"^┌ Error: Metal\.jl is only supported on macOS$", stop=r"^└ @ Metal .*$"),
)

const ANSI_ESCAPE_REGEX = r"\e\[[0-9;]*m"

function filter_stderr_content(text::AbstractString; rules=STDERR_SUPPRESSION_RULES)
    isempty(text) && return text
    lines = split(text, '\n'; keepempty=true)
    filtered = String[]
    active_stop = nothing
    for line in lines
        match_line = replace(line, ANSI_ESCAPE_REGEX => "")
        if !isnothing(active_stop)
            if occursin(active_stop, match_line)
                active_stop = nothing
            end
            continue
        end
        matched = false
        for rule in rules
            if occursin(rule.start, match_line)
                active_stop = rule.stop
                matched = true
                break
            end
        end
        matched || push!(filtered, line)
    end
    return join(filtered, '\n')
end

function import_with_filtered_stderr(modulename::Symbol; rules=PREIMPORT_STDERR_SUPPRESSION_RULES)
    mktemp() do path, io
        redirect_stderr(io) do
            @eval import $(modulename)
        end
        flush(io)
        close(io)
        filtered = filter_stderr_content(read(path, String); rules=rules)
        isempty(filtered) || print(Base.stderr, filtered)
    end
end

import ParallelStencil # Precompile it.
import ParallelStencil: SUPPORTED_PACKAGES, PKG_CUDA, PKG_AMDGPU, PKG_METAL, PKG_KERNELABSTRACTIONS
@static if (PKG_CUDA in SUPPORTED_PACKAGES) import CUDA end
@static if (PKG_AMDGPU in SUPPORTED_PACKAGES) import AMDGPU end
@static if (PKG_METAL in SUPPORTED_PACKAGES) import_with_filtered_stderr(:Metal) end
@static if (PKG_KERNELABSTRACTIONS in SUPPORTED_PACKAGES) import KernelAbstractions end # KernelAbstractions does not require extra harness env vars beyond the existing CUDA/AMDGPU settings.

excludedfiles = [ "test_excluded.jl", "test_incremental_compilation.jl", "test_revise.jl"]; # TODO: test_incremental_compilation has to be deactivated until Polyester support released

const STDERR_SUPPRESSION_RULES = (
    (name="metadata method overwrite warnings", start=r"^WARNING: Method definition .*###META.* overwritten.*$", stop=nothing),
    (name="[T]Data module replacement warnings", start=r"^WARNING: replacing module [T]?Data\.$", stop=nothing),
    (name="Metal OS support warnings", start=r"^┌ Error: Metal\.jl is only supported on macOS$", stop=r"^└ @ Metal .*$"),
)

function runtests(testfiles=String[]; stop_on_fail=false)
    exename   = joinpath(Sys.BINDIR, Base.julia_exename())
    testdir   = pwd()
    istest(f) = endswith(f, ".jl") && startswith(basename(f), "test_")
    testfiles = isempty(testfiles) ? sort(filter(istest, vcat([joinpath.(root, files) for (root, dirs, files) in walkdir(testdir)]...))) : testfiles

    nabort = 0
    nfail  = 0
    abortfiles = String[]
    failfiles  = String[]
    first_fail_file = ""
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
            print(Base.stderr, filter_stderr_content(stderr_content))
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
            if stop_on_fail
                first_fail_file = f
                break
            end
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
        elseif !stop_on_fail
            printstyled("Test suite: all selected test files executed, but some test file(s) have tests that failed or errored (see message above).\n"; bold=true, color=:red)
        end
    end
    if stop_on_fail && !isempty(first_fail_file)
        printstyled("Test suite: stopped at first test file with tests that failed or errored: $first_fail_file\n"; bold=true, color=:red)
    end
    println("")
    return nabort+nfail
end

stop_on_fail = any(==("--stop-on-fail"), ARGS)
testfiles = filter(!=("--stop-on-fail"), ARGS)
exit(runtests(testfiles; stop_on_fail=stop_on_fail))
