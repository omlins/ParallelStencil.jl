const ERRMSG_OVERLAP = "@overlap: the code block must only contain @parallel calls. Type `?@overlap` for more information."
const ERRMSG_INVALID_STREAM_OVERLAP = "@overlap: invalid keyword argument in @parallel <kernelcall>: stream. Specifying the stream is not allowed in a @parallel call inside the code block argument of @overlap."

const OVERLAP_DOC = """
    @overlap block

Overlap all `@parallel` calls in `block` by executing them asynchronously on separate streams and synchronizing them at the end of the block.

# Arguments
- `block`: code block containing multiple [`@parallel`](@ref) calls to be overlapped. Other code in the block, including [`@parallel_async`](@ref) or [`@synchronize`](@ref) calls, is not allowed (would lead to unclear semantics); any other code must be placed before or after the `@overlap` block. The [`@parallel`](@ref) calls cannot contain the stream keyword argument (stream=...); any other positional and keyword arguments are allowed.

# Examples
    @overlap begin
        @parallel kernel1!(array1)
        @parallel kernel2!(array2)
        @parallel kernel3!(array3)
    end

See also: [`@parallel`](@ref), [`@parallel_async`](@ref), [`@synchronize`](@ref)
"""
@doc OVERLAP_DOC
macro overlap(args...) check_initialized(__module__); checkargs_overlap(args...); esc(overlap(__module__, args...)); end


## ARGUMENT CHECKS

function checkargs_overlap(args...)
    if (length(args) != 1) @ArgumentError("wrong number of arguments.") end
    if !is_block(args[1]) @ArgumentError("the argument must be a code block (obtained: $(args[1])).") end
end

function overlap(caller::Module, args::Union{Symbol,Expr}...; package::Symbol=get_package(caller))
    checkargs_overlap(args...)
    block = args[1]
    if     isgpu(package) overlap_gpu(block)
    elseif iscpu(package) overlap_cpu(block)
    else                  @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") # ERRMSG_UNSUPPORTED_PACKAGE defined in shared.jl
    end
end

function overlap_gpu(block::Expr)
    calls = collect_parallel_calls(block)
    async_calls = Expr[]
    for (i, call) in enumerate(calls)
        push!(async_calls, :(@parallel_async $(call.posargs...) stream=ParallelStencil.ParallelKernel.@get_stream($i) $(call.kwargs...) $(call.kernelcall)))
    end
    sync_calls = [:(ParallelStencil.ParallelKernel.@synchronize(ParallelStencil.ParallelKernel.@get_stream($i))) for i in 1:length(async_calls)]
    quote
        $(async_calls...)
        $(sync_calls...)
    end
end

function collect_parallel_calls(block::Expr)
    statements = filter(x->!isa(x, LineNumberNode), block.args)
    if isempty(statements) @ArgumentError(ERRMSG_OVERLAP) end
    calls = []
    for (i, statement) in enumerate(statements)
        if !is_parallel_call(statement) @ArgumentError(ERRMSG_OVERLAP) end
        parallel_args = extract_args(statement, Symbol("@parallel"))
        posargs, kwargs, kernelcall = split_parallel_args(parallel_args)
        if any(x->(length(x.args) >= 1 && x.args[1] == :stream), kwargs) @ArgumentError(ERRMSG_INVALID_STREAM_OVERLAP) end
        push!(calls, (posargs=posargs, kwargs=kwargs, kernelcall=kernelcall))
    end
    return calls
end

function overlap_cpu(block::Expr)
    calls = collect_parallel_calls(block)
    seq_calls = [:(@parallel $(call.posargs...) $(call.kwargs...) $(call.kernelcall)) for call in calls]
    quote
        $(seq_calls...)
    end
end
