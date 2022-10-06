const ERRMSG_HIDE_COMMUNICATION = "@hide_communication: the code block must start with exactly one @parallel computation call and be followed by code to set boundary conditions and to perform communication. Type `?@hide_communication` for more information."
const ERRMSG_INVALID_STREAM = "@hide_communication: invalid keyword argument in @parallel <kernelcall>: stream. Specifying the stream is not allowed in a @parallel call inside the code block argument of @hide_communication."
const ERRMSG_INVALID_COMP = "@hide_communication: invalid positional arguments in @parallel <kernelcall>: no positional argument is allowed in the @parallel call to perform computations inside the code block argument of @hide_communication."
const ERRMSG_INVALID_BC_COMM = "@hide_communication: invalid positional arguments in @parallel <kernelcall>: exactly one positional argument argument (ranges) must be passed in @parallel calls to set boundary conditions and to perform communication inside the code block argument of @hide_communication."

##
const HIDE_COMMUNICATION_DOC = """
    @hide_communication boundary_width block

!!! note "Advanced"
        @hide_communication ranges_outer ranges_inner block

Hide the communication behind the computation within the code `block`.

# Arguments
- `boundary_width::Tuple{Integer,Integer,Integer} | Tuple{Integer,Integer} | Tuple{Integer}`: width of the boundaries in each dimension. The boundaries must include (at least) all the data that is accessed in the communcation performed.
- `block`: code block wich starts with exactly one [`@parallel`](@ref) call to perform computations, followed by code to set boundary conditions and to perform communication (as e.g. `update_halo!` from the package `ImplicitGlobalGrid`). The [`@parallel`](@ref) call to perform computations cannot contain any positional arguments (ranges, nblocks or nthreads) nor the stream keyword argument (stream=...). The code to set boundary conditions and to perform communication must only access the elements in the boundary ranges of the fields modified in the [`@parallel`](@ref) call; all elements can be acccessed from other fields. Moreover, this code must not include statements in array broadcasting notation, because they are always run on the default stream in CUDA (for CUDA.jl < v2.0), which makes CUDA stream overlapping impossible. Instead, boundary region elements can, e.g., be accessed with [`@parallel`](@ref) calls passing a ranges argument that ensures that no threads mapping to elements outside of `ranges_outer` are launched. Note that these [`@parallel`](@ref) `ranges` calls cannot contain any other positional arguments (nblocks or nthreads) nor the stream keyword argument (stream=...).

!!! note "Advanced"
    - `ranges_outer::`Tuple with one or multiple `ranges` as required by the corresponding argument of [`@parallel`](@ref): the `ranges` must together span (at least) all the data that is accessed in the communcation and boundary conditions performed.
    - `ranges_inner::`Tuple with one or multiple `ranges` as required by the corresponding argument of [`@parallel`](@ref): the `ranges` must together span the data that is not included by `ranges_outer`.

# Examples
    @hide_communication (16, 2, 2) begin
        @parallel diffusion3D_step!(Te2, Te, Ci, lam, dt, dx, dy, dz);
        update_halo!(Te2);
    end

    @hide_communication (16, 2) begin
        @parallel diffusion2D_step!(Te2, Te, Ci, lam, dt, dx, dy);
        update_halo!(Te2);
    end

    @hide_communication ranges_outer ranges_inner begin
        @parallel diffusion3D_step!(Te2, Te, Ci, lam, dt, dx, dy, dz);
        update_halo!(Te2);
    end

    @parallel_indices (iy,iz) function bc_x(A)
        A[  1, iy,  iz] = A[    2,   iy,   iz]
        A[end, iy,  iz] = A[end-1,   iy,   iz]
        return
    end
    @parallel_indices (ix,iz) function bc_y(A)
        A[ ix,  1,  iz] = A[   ix,    2,   iz]
        A[ ix,end,  iz] = A[   ix,end-1,   iz]
        return
    end
    @parallel_indices (ix,iy) function bc_z(A)
        A[ ix,  iy,  1] = A[   ix,   iy,    2]
        A[ ix,  iy,end] = A[   ix,   iy,end-1]
        return
    end
    @hide_communication (16, 2, 2) begin
        @parallel diffusion3D_step!(Te2, Te, Ci, lam, dt, dx, dy, dz);
        @parallel (1:size(Te,2), 1:size(Te,3)) bc_x(Te);
        @parallel (1:size(Te,1), 1:size(Te,3)) bc_y(Te);
        @parallel (1:size(Te,1), 1:size(Te,2)) bc_z(Te);
        update_halo!(Te2);
    end

!!! note "Developers note"
    The communcation should not perform any blocking operations to enable a maximal overlap of communication with computation.

See also: [`@parallel`](@ref)
"""
@doc HIDE_COMMUNICATION_DOC
macro hide_communication(args...) check_initialized(); checkargs_hide_communication(args...); esc(hide_communication(args...)); end

macro get_priority_stream(args...) check_initialized(); checkargs_get_stream(args...); esc(get_priority_stream(args...)); end
macro get_stream(args...) check_initialized(); checkargs_get_stream(args...); esc(get_stream(args...)); end


## ARGUMENT CHECKS

function checkargs_hide_communication(args...)
    if (length(args) < 2 || length(args) > 3) @ArgumentError("wrong number of arguments.") end
    if !is_block(args[end]) @ArgumentError("the last argument must be a code block (obtained: $(args[end])).") end
end

function checkargs_get_stream(args...)
    if (length(args) != 1) @ArgumentError("wrong number of arguments.") end
end


## GATEWAY FUNCTIONS

function hide_communication(args::Union{Integer,Symbol,Expr}...; package::Symbol=get_package())
    if     (package == PKG_CUDA)    hide_communication_gpu(args...)
    elseif (package == PKG_THREADS) hide_communication_threads(args...)
    else                            @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function get_priority_stream(args::Union{Integer,Symbol,Expr}...; package::Symbol=get_package())
    if     (package == PKG_CUDA)    get_priority_stream_cuda(args...)
    elseif (package == PKG_AMDGPU)  get_priority_stream_amdgpu(args...)
    else                            @ArgumentError("unsupported GPU package (obtained: $package).")
    end
end

function get_stream(args::Union{Integer,Symbol,Expr}...; package::Symbol=get_package())
    if     (package == PKG_CUDA)    get_stream_cuda(args...)
    elseif (package == PKG_AMDGPU)  get_stream_amdgpu(args...)
    else                            @ArgumentError("unsupported GPU package (obtained: $package).")
    end
end

## @HIDE_COMMUNICATION FUNCTIONS

function hide_communication_gpu(ranges_outer::Union{Symbol,Expr}, ranges_inner::Union{Symbol,Expr}, block::Expr)
    compcall, bc_and_commcalls = extract_calls(block)
    parallel_args = extract_args(compcall, Symbol("@parallel"))
    posargs, kwargs, compkernelcall = split_parallel_args(parallel_args)
    if any([x.args[1]==:stream for x in kwargs]) @ArgumentError(ERRMSG_INVALID_STREAM) end
    if (length(posargs) > 0) @ArgumentError(ERRMSG_INVALID_COMP) end
    bc_and_commcalls = process_bc_and_commcalls(bc_and_commcalls)
    quote
        for i in 1:length($ranges_outer)
            @parallel_async $ranges_outer[i] stream=ParallelStencil.ParallelKernel.@get_priority_stream(i) $(kwargs...) $compkernelcall #NOTE: it cannot directly go to ParallelStencil.ParallelKernel.@parallel_async as else it cannot honour ParallelStencil args as loopopt (fixing it to ParallelStencil is also not possible as it assumes, else the ParalellKernel hide_communication unit tests fail).
        end
        for i in 1:length($ranges_inner)
            @parallel_async $ranges_inner[i] stream=ParallelStencil.ParallelKernel.@get_stream(i) $(kwargs...) $compkernelcall          #NOTE: ...
        end
        for i in 1:length($ranges_outer) ParallelStencil.ParallelKernel.@synchronize(ParallelStencil.ParallelKernel.@get_priority_stream(i)); end
        $bc_and_commcalls
        for i in 1:length($ranges_inner) ParallelStencil.ParallelKernel.@synchronize(ParallelStencil.ParallelKernel.@get_stream(i)); end
    end
end

function hide_communication_threads(ranges_outer::Union{Symbol,Expr}, ranges_inner::Union{Symbol,Expr}, block::Expr)
    return block #NOTE: This is currently not implemented, but enables correct execution nevertheless.
end


function hide_communication_gpu(boundary_width::Union{Integer,Symbol,Expr}, block::Expr)
    compcall, = extract_calls(block)
    parallel_args = extract_args(compcall, Symbol("@parallel"))
    posargs, kwargs, compkernelcall = split_parallel_args(parallel_args)
    if any([x.args[1]==:stream for x in kwargs]) @ArgumentError(ERRMSG_INVALID_STREAM) end
    if (length(posargs) > 0) @ArgumentError(ERRMSG_INVALID_COMP) end
    ranges = :(ParallelStencil.ParallelKernel.get_ranges($(compkernelcall.args[2:end]...)))
    ranges_outer = gensym_world("ranges_outer", @__MODULE__)
    ranges_inner = gensym_world("ranges_inner", @__MODULE__)
    quote
        $ranges_outer = ParallelStencil.ParallelKernel.get_ranges_outer($boundary_width, $ranges)
        $ranges_inner = ParallelStencil.ParallelKernel.get_ranges_inner($boundary_width, $ranges)
        ParallelStencil.ParallelKernel.@hide_communication $ranges_outer $ranges_inner $block  # NOTE: to enable macros like @hide_communication_cuda, one would need to use everywhere in hide_communication_cuda _cuda and _thread suffixed macros and enable their processing. This is more complex and generates less readable code though. => Design decision: only enable _cuda/_threads macros for macros that do not return package dependent macros.
    end
end

function hide_communication_threads(boundary_width::Union{Integer,Symbol,Expr}, block::Expr)
    return block #NOTE: This is currently not implemented, but enables correct execution nevertheless.
end


# @GET_STREAM AND @GET_PRIORITY_STEAM FUNCTIONS

get_priority_stream_cuda(id::Union{Integer,Symbol,Expr})   = return :(ParallelStencil.ParallelKernel.get_priority_custream($id))
get_priority_stream_amdgpu(id::Union{Integer,Symbol,Expr}) = return :(ParallelStencil.ParallelKernel.get_priority_rocstream($id))
get_stream_cuda(id::Union{Integer,Symbol,Expr})            = return :(ParallelStencil.ParallelKernel.get_custream($id))
get_stream_amdgpu(id::Union{Integer,Symbol,Expr})          = return :(ParallelStencil.ParallelKernel.get_rocstream($id))


## FUNCTIONS TO EXTRACT AND PROCESS COMPUTATION AND BOUNDARY CONDITIONS CALLS / COMMUNICATION CALLS

function extract_calls(block::Expr)
    if (sum([isa(arg, Expr) for arg in block.args]) < 2) @ArgumentError(ERRMSG_HIDE_COMMUNICATION) end
    compcall         = block.args[2]
    bc_and_commcalls = quote $(block.args[3:end]...) end
    if !is_parallel_call(compcall) @ArgumentError(ERRMSG_HIDE_COMMUNICATION) end
    return compcall, bc_and_commcalls
end

function process_bc_and_commcalls(block::Expr)
    return postwalk(block) do x
        if !is_parallel_call(x) return x; end
        @capture(x, @parallel args__) || @ModuleInternalError("a @parallel call could not be pattern matched.")
        posargs, kwargs = split_parallel_args(args)
        if any([x.args[1]==:stream for x in kwargs]) @ArgumentError(ERRMSG_INVALID_STREAM) end
        if (length(posargs) != 1) @ArgumentError(ERRMSG_INVALID_BC_COMM) end
        return :(@parallel $(args[1:end-1]...) stream = ParallelStencil.ParallelKernel.get_priority_custream(1) $(args[end]))
    end
end


## FUNCTIONS TO GET INNER AND OUTER RANGES AND TO PROMOTE BOUNDARY_WIDTH TO 3D

promote_boundary_width(boundary_width::BOUNDARY_WIDTH_TYPE_1D)       = (boundary_width,    0, 0)
promote_boundary_width(boundary_width::BOUNDARY_WIDTH_TYPE_1D_TUPLE) = (boundary_width..., 0, 0)
promote_boundary_width(boundary_width::BOUNDARY_WIDTH_TYPE_2D)       = (boundary_width..., 0)
promote_boundary_width(boundary_width::BOUNDARY_WIDTH_TYPE)          = boundary_width
promote_boundary_width(boundary_width)                               = @ArgumentError("@hide_communication: boundary_width must be a Tuple of Integer of size 1, 2 or 3 (obtained: $boundary_width; its type is: $(typeof(boundary_width))).")

function get_ranges_outer(boundary_width, ranges::RANGES_TYPE) where T <:Integer
    boundary_width = promote_boundary_width(boundary_width)
    validate_ranges_args(boundary_width, ranges)
    ms = length.(ranges)
    bw = boundary_width
    if ms[3] > 1 # 3D
        ranges_outer = ((1:ms[1],             1:ms[2],             1:bw[3]),
                        (1:ms[1],             1:ms[2],             ms[3]-bw[3]+1:ms[3]),
                        (1:ms[1],             1:bw[2],             bw[3]+1:ms[3]-bw[3]),
                        (1:ms[1],             ms[2]-bw[2]+1:ms[2], bw[3]+1:ms[3]-bw[3]),
                        (1:bw[1],             bw[2]+1:ms[2]-bw[2], bw[3]+1:ms[3]-bw[3]),
                        (ms[1]-bw[1]+1:ms[1], bw[2]+1:ms[2]-bw[2], bw[3]+1:ms[3]-bw[3]),
                       )
    elseif ms[2] > 1 # 2D
        ranges_outer = ((1:ms[1],             1:bw[2],             1:1),
                        (1:ms[1],             ms[2]-bw[2]+1:ms[2], 1:1),
                        (1:bw[1],             bw[2]+1:ms[2]-bw[2], 1:1),
                        (ms[1]-bw[1]+1:ms[1], bw[2]+1:ms[2]-bw[2], 1:1),
                       )
    elseif ms[1] > 1  # 1D
        ranges_outer = ((1:bw[1],             1:1, 1:1),
                        (ms[1]-bw[1]+1:ms[1], 1:1, 1:1),
                       )
    else
        @ModuleInternalError("invalid argument 'ranges'.")
    end
    return Tuple([r for r in ranges_outer if all(length.(r) .!= 0)])
end

function get_ranges_inner(boundary_width, ranges::RANGES_TYPE) where T <:Integer
    boundary_width = promote_boundary_width(boundary_width)
    validate_ranges_args(boundary_width, ranges)
    ms = length.(ranges)
    bw = boundary_width
    if     (ms[3] > 1) return ( (bw[1]+1:ms[1]-bw[1], bw[2]+1:ms[2]-bw[2], bw[3]+1:ms[3]-bw[3]), ) # 3D
    elseif (ms[2] > 1) return ( (bw[1]+1:ms[1]-bw[1], bw[2]+1:ms[2]-bw[2], 1:1), )                 # 2D
    elseif (ms[1] > 1) return ( (bw[1]+1:ms[1]-bw[1], 1:1, 1:1), )                                 # 1D
    else @ModuleInternalError("invalid argument 'ranges'.")
    end
end

function validate_ranges_args(boundary_width::Tuple{T,T,T}, ranges::RANGES_TYPE) where T <:Integer
    maxsize = length.(ranges)
    if any([r.start != 1 for r in ranges]) @ArgumentError("unsupported call to `@hide_communication boundary_width block`: automatic ranges determination for @hide_communication is not supported for ranges with start != 1. Use the explicit version `@hide_communication ranges_outer ranges_inner block` instead.") end
    if !all(maxsize .> 0) @IncoherentArgumentError("incoherent arguments in @hide_communication: the following must be true: all(length.(ranges) .> 0) ") end
    if !all(maxsize .> 2 .* boundary_width) @IncoherentArgumentError("incoherent arguments in @hide_communication: the following must be true: all(length.(ranges) .> 2 .* boundary_width) ") end # NOTE: this ensures among other things that if maxsize == 1, then boundary_width==0.
    if any((maxsize .> 1) .& (boundary_width .== 0)) @IncoherentArgumentError("incoherent arguments in @hide_communication: the following must be false: any((length.(ranges) .> 1) .& (boundary_width .== 0)) ") end
    if all(maxsize .== 1) @IncoherentArgumentError("incoherent arguments in @hide_communication: the following must be false: all(length.(ranges) .== 1)") end
end
