const ERRMSG_HIDE_COMMUNICATION = "@hide_communication: the code block must normally start with one @parallel computation call (for exceptions, see keyword `computation_calls`) and be followed by code to set boundary conditions and to perform communication. Type `?@hide_communication` for more information."
const ERRMSG_INVALID_STREAM = "@hide_communication: invalid keyword argument in @parallel <kernelcall>: stream. Specifying the stream is not allowed in a @parallel call inside the code block argument of @hide_communication."
const ERRMSG_INVALID_COMP = "@hide_communication: invalid positional arguments in @parallel <kernelcall>: zero or one positional argument (ranges) is allowed in the @parallel call to perform computations inside the code block argument of @hide_communication <boundary_width> <block>."
const ERRMSG_INVALID_COMP_2 = "@hide_communication: invalid positional arguments in @parallel <kernelcall>: no positional argument is allowed in the @parallel call to perform computations inside the code block argument of @hide_communication <ranges_outer> <ranges_inner> <block>."
const ERRMSG_INVALID_BC_COMM = "@hide_communication: invalid positional arguments in @parallel <kernelcall>: exactly one positional argument argument (ranges) must be passed in @parallel calls to set boundary conditions and to perform communication inside the code block argument of @hide_communication."

##
const HIDE_COMMUNICATION_DOC = """
    @hide_communication boundary_width block

!!! note "Advanced"
        @hide_communication boundary_width block computation_calls=...
        @hide_communication ranges_outer ranges_inner block
        @hide_communication ranges_outer ranges_inner block computation_calls=...

Hide the communication behind the computation within the code `block`.

# Arguments
- `boundary_width::Tuple{Integer,Integer,Integer} | Tuple{Integer,Integer} | Tuple{Integer}`: width of the boundaries in each dimension. The boundaries must include (at least) all the data that is accessed in the communication performed.
- `block`: code block wich starts with one [`@parallel`](@ref) call to perform computations (for exceptions, see keyword `computation_calls`), followed by code to set boundary conditions and to perform communication (as e.g. `update_halo!` from the package `ImplicitGlobalGrid`). The [`@parallel`](@ref) call to perform computations can normally not contain any positional arguments (ranges, nblocks or nthreads) nor the stream keyword argument (stream=...) (for exceptions, see keyword `computation_calls`). The code to set boundary conditions and to perform communication must only access the elements in the boundary ranges of the fields modified in the [`@parallel`](@ref) call; all elements can be acccessed from other fields. Moreover, this code must not include statements in array broadcasting notation, because they are always run on the default stream in CUDA (for CUDA.jl < v2.0), which makes CUDA stream overlapping impossible. Instead, boundary region elements can, e.g., be accessed with [`@parallel`](@ref) calls passing a ranges argument that ensures that no threads mapping to elements outside of the boundary regions are launched. Note that these [`@parallel`](@ref) `ranges` calls cannot contain any other positional arguments (nblocks or nthreads) nor the stream keyword argument (stream=...).

!!! note "Advanced"
    - `ranges_outer::`Tuple with one or multiple `ranges` as required by the corresponding argument of [`@parallel`](@ref): the `ranges` must together span (at least) all the data that is accessed in the communication and boundary conditions performed.
    - `ranges_inner::`Tuple with one or multiple `ranges` as required by the corresponding argument of [`@parallel`](@ref): the `ranges` must together span the data that is not included by `ranges_outer`.

!!! note "Advanced keyword arguments"
    - `computation_calls=1::`number of [`@parallel`](@ref) calls at the start of the code `block` that contain the computation to be hidden behind communication. Only set this argument if you are sure that it is safe to deviate from the standard behavior (`computation_calls=1`). Note that, in many scenarios (slightly) wrong results will be obtained when `computation_calls` in greater than `1`; it is only safe, generally speaking, if no point in the boundary region computations depends on any inner point computation (finite difference derivatives can, for example, constitute such a dependency leading to wrong results).

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

    @hide_communication (16, 2, 2) begin
        @parallel (1:size(Te,1), 1:size(Te,2), 1:size(Te,3)) diffusion3D_step!(Te2, Te, Ci, lam, dt, dx, dy, dz);
        @parallel (1:size(Te,2), 1:size(Te,3)) bc_x(Te);
        @parallel (1:size(Te,1), 1:size(Te,3)) bc_y(Te);
        @parallel (1:size(Te,1), 1:size(Te,2)) bc_z(Te);
        update_halo!(Te2);
    end

    @hide_communication (16, 2, 2) computation_calls=2 begin
        @parallel computation1!(A2, A, B);
        @parallel computation2!(B, C);
        @parallel (1:size(A,2), 1:size(A,3)) bc_x(A);
        @parallel (1:size(A,1), 1:size(A,3)) bc_y(A);
        @parallel (1:size(A,1), 1:size(A,2)) bc_z(A);
        update_halo!(Te2);
    end

!!! note "Developers note"
    The communication should not perform any blocking operations to enable a maximal overlap of communication with computation.

See also: [`@parallel`](@ref)
"""
@doc HIDE_COMMUNICATION_DOC
macro hide_communication(args...) check_initialized(__module__); checkargs_hide_communication(args...); esc(hide_communication(__module__, args...)); end

const GET_PRIORITY_STREAM_DOC = """
    @get_priority_stream(id)

Get the priority stream with identifier `id` for the package selected with [`@init_parallel_kernel`](@ref). Returns `nothing` for CPU backends.

If no stream with the given identifier exists yet, then a new stream is created for this identifier. All stream handles are stored package-internally and can be re-retrieved anytime at no cost.

If no stream with the given identifier exists yet, then a new stream is created for this identifier. All stream handles are stored package-internally and can be re-retrieved anytime at no cost.

# Arguments
- `id::Integer`: identifier of the stream

See also: [`@parallel_async`](@ref), [`@synchronize`](@ref)
"""
@doc GET_PRIORITY_STREAM_DOC
macro get_priority_stream(args...)
    check_initialized(__module__);
    checkargs_get_stream(args...);
    esc(get_priority_stream(__module__, args...));
end

const GET_STREAM_DOC = """
    @get_stream(id)

Get the default-priority stream with identifier `id` for the package selected with [`@init_parallel_kernel`](@ref). Returns `nothing` for CPU backends.

If no stream with the given identifier exists yet, then a new stream is created for this identifier. All stream handles are stored package-internally and can be re-retrieved anytime at no cost.

If no stream with the given identifier exists yet, then a new stream is created for this identifier. All stream handles are stored package-internally and can be re-retrieved anytime at no cost.

# Arguments
- `id::Integer`: identifier of the stream

See also: [`@parallel_async`](@ref), [`@synchronize`](@ref)
"""
@doc GET_STREAM_DOC
macro get_stream(args...)
    check_initialized(__module__);
    checkargs_get_stream(args...);
    esc(get_stream(__module__, args...));
end


## ARGUMENT CHECKS

function checkargs_hide_communication(args...)
    posargs, ~ = split_args(args)
    if (length(posargs) < 2 || length(posargs) > 3) @ArgumentError("wrong number of positional arguments.") end
    if !is_block(args[end]) @ArgumentError("the last argument must be a code block (obtained: $(args[end])).") end
end

function checkargs_get_stream(args...)
    if (length(args) != 1) @ArgumentError("wrong number of arguments.") end
end


## GATEWAY FUNCTIONS

function hide_communication(caller::Module, args::Union{Integer,Symbol,Expr}...; package::Symbol=get_package(caller))
    posargs, kwargs_expr = split_args(args)
    kwargs, ~ = extract_kwargs(caller, kwargs_expr, (:computation_calls,), "@hide_communication", false; eval_args=(:computation_calls,))
    if     isgpu(package) hide_communication_gpu(posargs...; kwargs...)
    elseif iscpu(package) hide_communication_cpu(posargs...; kwargs...)
    else                  @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
end

function get_priority_stream(caller::Module, args::Union{Integer,Symbol,Expr}...; package::Symbol=get_package(caller))
    if     (package == PKG_CUDA)    get_priority_stream_cuda(args...)
    elseif (package == PKG_AMDGPU)  get_priority_stream_amdgpu(args...)
    elseif (package == PKG_METAL)   get_priority_stream_metal(args...)
    elseif iscpu(package)           :(nothing)
    else                            @ArgumentError("unsupported package type (obtained: $package).")
    end
end

function get_stream(caller::Module, args::Union{Integer,Symbol,Expr}...; package::Symbol=get_package(caller))
    if     (package == PKG_CUDA)    get_stream_cuda(args...)
    elseif (package == PKG_AMDGPU)  get_stream_amdgpu(args...)
    elseif (package == PKG_METAL)   get_stream_metal(args...)
    elseif iscpu(package)           :(nothing)
    else                            @ArgumentError("unsupported package type (obtained: $package).")
    end
end

## @HIDE_COMMUNICATION FUNCTIONS

function hide_communication_gpu(ranges_outer::Union{Symbol,Expr}, ranges_inner::Union{Symbol,Expr}, block::Expr; computation_calls::Integer=1)
    if (computation_calls < 1) @KeywordArgumentError("Invalid keyword argument in @hide_communication: computation_calls must be >= 1.") end
    compcalls, bc_and_commcalls = extract_calls(block, computation_calls)
    compcalls_outer = []
    compcalls_inner = []
    for compcall in compcalls
        parallel_args = extract_args(compcall, Symbol("@parallel"))
        posargs, kwargs, compkernelcall = split_parallel_args(parallel_args)
        if any([x.args[1]==:stream for x in kwargs]) @ArgumentError(ERRMSG_INVALID_STREAM) end
        if (length(posargs) > 0) @ArgumentError(ERRMSG_INVALID_COMP_2) end
        push!(compcalls_outer, :(@parallel_async $ranges_outer[i] stream=ParallelStencil.ParallelKernel.@get_priority_stream(i) $(kwargs...) $compkernelcall)) #NOTE: it cannot directly go to ParallelStencil.ParallelKernel.@parallel_async as else it cannot honour ParallelStencil args as memopt (fixing it to ParallelStencil is also not possible as it assumes, else the ParalellKernel hide_communication unit tests fail).
        push!(compcalls_inner, :(@parallel_async $ranges_inner[i] stream=ParallelStencil.ParallelKernel.@get_stream(i) $(kwargs...) $compkernelcall))          #NOTE: ...
    end
    bc_and_commcalls = flatten(process_bc_and_commcalls(bc_and_commcalls))
    if comm_is_splitted(bc_and_commcalls)
        bc_and_commcalls_z, bc_and_commcalls_xy = split_bc_and_commcalls(bc_and_commcalls)
        quote
            for i in 1:length($ranges_outer)
                $(compcalls_outer...)
            end
            for i in 2:3 ParallelStencil.ParallelKernel.@synchronize(ParallelStencil.ParallelKernel.@get_priority_stream(i)); end # NOTE: synchronize the streams of the z-boundary computations (assumed to be stream 2 and 3 - to be in agreement with get_ranges_outer)
            $bc_and_commcalls_z
            for i in 1:length($ranges_inner)
                $(compcalls_inner...)
            end
            for i in 1:1 ParallelStencil.ParallelKernel.@synchronize(ParallelStencil.ParallelKernel.@get_priority_stream(i)); end
            for i in 4:length($ranges_outer) ParallelStencil.ParallelKernel.@synchronize(ParallelStencil.ParallelKernel.@get_priority_stream(i)); end
            $bc_and_commcalls_xy
            for i in 1:length($ranges_inner) ParallelStencil.ParallelKernel.@synchronize(ParallelStencil.ParallelKernel.@get_stream(i)); end
        end
    else
        quote
            for i in 1:length($ranges_outer)
                $(compcalls_outer...)
            end
            for i in 1:length($ranges_inner)
                $(compcalls_inner...)
            end
            for i in 1:length($ranges_outer) ParallelStencil.ParallelKernel.@synchronize(ParallelStencil.ParallelKernel.@get_priority_stream(i)); end
            $bc_and_commcalls
            for i in 1:length($ranges_inner) ParallelStencil.ParallelKernel.@synchronize(ParallelStencil.ParallelKernel.@get_stream(i)); end
        end
    end
end

function hide_communication_cpu(ranges_outer::Union{Symbol,Expr}, ranges_inner::Union{Symbol,Expr}, block::Expr; computation_calls::Integer=1)
    return block #NOTE: This is currently not implemented, but enables correct execution nevertheless.
end


function hide_communication_gpu(boundary_width::Union{Integer,Symbol,Expr}, block::Expr; computation_calls::Integer=1)
    if (computation_calls < 1) @KeywordArgumentError("Invalid keyword argument in @hide_communication: computation_calls must be >= 1.") end
    compcalls, bc_and_commcalls = extract_calls(block, computation_calls)

    USE_EXPERIMENTAL = false
    if (USE_EXPERIMENTAL) bc_and_commcalls = flatten(split_commcalls(bc_and_commcalls)) end

    compranges = []
    for i in 1:length(compcalls)
        parallel_args = extract_args(compcalls[i], Symbol("@parallel"))
        posargs, kwargs, compkernelcall = split_parallel_args(parallel_args)
        if any([x.args[1]==:stream for x in kwargs]) @ArgumentError(ERRMSG_INVALID_STREAM) end
        if (length(posargs) > 1) @ArgumentError(ERRMSG_INVALID_COMP) end
        ranges = (length(posargs) == 1) ? posargs[1] : :(ParallelStencil.ParallelKernel.get_ranges($(compkernelcall.args[2:end]...)))
        push!(compranges, ranges)
        compcalls[i] = :(@parallel $(kwargs...) $compkernelcall)
    end
    ranges_outer = gensym_world("ranges_outer", @__MODULE__)
    ranges_inner = gensym_world("ranges_inner", @__MODULE__)
    quote
        $ranges_outer = ParallelStencil.ParallelKernel.get_ranges_outer($boundary_width, $(compranges...))
        $ranges_inner = ParallelStencil.ParallelKernel.get_ranges_inner($boundary_width, $(compranges...))
        ParallelStencil.ParallelKernel.@hide_communication $ranges_outer $ranges_inner computation_calls=$computation_calls begin  # NOTE: to enable macros like @hide_communication_cuda, one would need to use everywhere in hide_communication_cuda _cuda and _thread suffixed macros and enable their processing. This is more complex and generates less readable code though. => Design decision: only enable _cuda/_threads macros for macros that do not return package dependent macros.
            $(compcalls...)
            $(unblock(bc_and_commcalls))
        end
    end
end

function hide_communication_cpu(boundary_width::Union{Integer,Symbol,Expr}, block::Expr; computation_calls::Integer=1)
    return block #NOTE: This is currently not implemented, but enables correct execution nevertheless.
end


# @GET_STREAM AND @GET_PRIORITY_STEAM FUNCTIONS

get_priority_stream_cuda(id::Union{Integer,Symbol,Expr})   = return :(ParallelStencil.ParallelKernel.get_priority_custream($id))
get_priority_stream_amdgpu(id::Union{Integer,Symbol,Expr}) = return :(ParallelStencil.ParallelKernel.get_priority_rocstream($id))
get_priority_stream_metal(id::Union{Integer,Symbol,Expr})  = return :(ParallelStencil.ParallelKernel.get_priority_metalstream($id))
get_stream_cuda(id::Union{Integer,Symbol,Expr})            = return :(ParallelStencil.ParallelKernel.get_custream($id))
get_stream_amdgpu(id::Union{Integer,Symbol,Expr})          = return :(ParallelStencil.ParallelKernel.get_rocstream($id))
get_stream_metal(id::Union{Integer,Symbol,Expr})           = return :(ParallelStencil.ParallelKernel.get_metalstream($id))


## FUNCTIONS TO EXTRACT AND PROCESS COMPUTATION AND BOUNDARY CONDITIONS CALLS / COMMUNICATION CALLS

function extract_calls(block::Expr, computation_calls::Integer)
    if (sum([isa(arg, Expr) for arg in block.args]) < 1+computation_calls) @ArgumentError(ERRMSG_HIDE_COMMUNICATION) end
    statements       = filter(x->!isa(x, LineNumberNode), block.args)
    compcalls        = statements[1:computation_calls]
    bc_and_commcalls = quote $(statements[computation_calls+1:end]...) end
    if !all(is_parallel_call.(compcalls)) @ArgumentError(ERRMSG_HIDE_COMMUNICATION) end
    return compcalls, bc_and_commcalls
end

function process_bc_and_commcalls(block::Expr)
    return postwalk(block) do x
        if !is_parallel_call(x) return x; end
        @capture(x, @parallel args__) || @ModuleInternalError("a @parallel call could not be pattern matched.")
        posargs, kwargs = split_parallel_args(args)
        if any([x.args[1]==:stream for x in kwargs]) @ArgumentError(ERRMSG_INVALID_STREAM) end
        if (length(posargs) != 1) @ArgumentError(ERRMSG_INVALID_BC_COMM) end
        return :(@parallel $(args[1:end-1]...) stream = ParallelStencil.ParallelKernel.@get_priority_stream(1) $(args[end]))
    end
end

function split_commcalls(block::Expr)
    return postwalk(block) do x
        if !(!@capture(x, f_(args__; kwargs__)) && @capture(x, f_(args__)) && f == :update_halo!) return x; end
        return :(update_halo!($(args...); dims=(3,)); update_halo!($(args...); dims=(1,2)))
    end
end

function comm_is_splitted(block::Expr)
    if !is_block(block) return false; end
    statements = block.args
    has_comm_z = false
    has_comm_xy = false
    for statement in statements
        if @capture(statement, f_(args__; kwarg_))
            if     !has_comm_z && @capture(kwarg, dims=(3,))  has_comm_z = true
            elseif  has_comm_z && @capture(kwarg, dims=(1,2)) has_comm_xy = true
            end
        end
    end
    return has_comm_z && has_comm_xy
end

function split_bc_and_commcalls(block::Expr)
    if !is_block(block) @ModuleInternalError("expression is not a block; a block with at least two statements for communication is expected (obtained: $block)") end
    statements = block.args
    comm_z_pos  = -1
    for i in length(statements):-1:1
        if (@capture(statements[i], f_(args__; kwarg_)) && @capture(kwarg, dims=(3,))) comm_z_pos = i; break; end
    end
    if (comm_z_pos < 1) @ModuleInternalError("no communication statement with dims=(3,) found in the block.") end
    bc_and_commcalls_z  = quote $(statements[1:comm_z_pos]...) end
    bc_and_commcalls_xy = quote $(statements[comm_z_pos+1:end]...) end
    return bc_and_commcalls_z, bc_and_commcalls_xy
end


## FUNCTIONS TO GET INNER AND OUTER RANGES AND TO PROMOTE BOUNDARY_WIDTH TO 3D

promote_boundary_width(boundary_width::BOUNDARY_WIDTH_TYPE_1D)       = (boundary_width,    0, 0)
promote_boundary_width(boundary_width::BOUNDARY_WIDTH_TYPE_1D_TUPLE) = (boundary_width..., 0, 0)
promote_boundary_width(boundary_width::BOUNDARY_WIDTH_TYPE_2D)       = (boundary_width..., 0)
promote_boundary_width(boundary_width::BOUNDARY_WIDTH_TYPE)          = boundary_width
promote_boundary_width(boundary_width)                               = @ArgumentError("@hide_communication: boundary_width must be a Tuple of Integer of size 1, 2 or 3 (obtained: $boundary_width; its type is: $(typeof(boundary_width))).")

function determine_common_ranges(ranges::RANGES_TYPE...)
    if (length(ranges) > 1) && !all(map(x->x==ranges[1], ranges)) @ArgumentError("@hide_communication: the ranges of the computation calls are not all equal (obtained: $ranges). Make them equal setting the ranges manually or remove incompatible computation calls.") end
    ranges = ranges[1]
end

function get_ranges_outer(boundary_width, ranges::RANGES_TYPE...)
    ranges = determine_common_ranges(ranges...)
    boundary_width = promote_boundary_width(boundary_width)
    validate_ranges_args(boundary_width, ranges)
    ms = length.(ranges)
    bw = boundary_width
    if ms[3] > 1 # 3D
        ranges_outer = (
                        (1:bw[1],             bw[2]+1:ms[2]-bw[2], bw[3]+1:ms[3]-bw[3]), # 5
                        (1:ms[1],             1:ms[2],             1:bw[3]),             # 1
                        (1:ms[1],             1:ms[2],             ms[3]-bw[3]+1:ms[3]), # 2
                        (ms[1]-bw[1]+1:ms[1], bw[2]+1:ms[2]-bw[2], bw[3]+1:ms[3]-bw[3]), # 6
                        (1:ms[1],             1:bw[2],             bw[3]+1:ms[3]-bw[3]), # 3
                        (1:ms[1],             ms[2]-bw[2]+1:ms[2], bw[3]+1:ms[3]-bw[3]), # 4
                       )
    elseif ms[2] > 1 # 2D
        ranges_outer = (
                        (ms[1]-bw[1]+1:ms[1], bw[2]+1:ms[2]-bw[2], 1:1), # 4
                        (1:ms[1],             1:bw[2],             1:1), # 1
                        (1:ms[1],             ms[2]-bw[2]+1:ms[2], 1:1), # 2
                        (1:bw[1],             bw[2]+1:ms[2]-bw[2], 1:1), # 3
                       )
    elseif ms[1] > 1  # 1D
        ranges_outer = (
                        (ms[1]-bw[1]+1:ms[1], 1:1, 1:1), # 2
                        (1:bw[1],             1:1, 1:1), # 1
                       )
    else
        @ModuleInternalError("invalid argument 'ranges'.")
    end
    return Tuple([r for r in ranges_outer if all(length.(r) .!= 0)])
end

function get_ranges_inner(boundary_width, ranges::RANGES_TYPE...)
    ranges = determine_common_ranges(ranges...)
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
