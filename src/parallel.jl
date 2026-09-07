import .ParallelKernel: get_name, set_name, get_body, set_body!, add_return, remove_return, extract_kwargs, split_parallel_args, extract_tuple, substitute, literaltypes, push_to_signature!, add_loop, add_threadids, promote_maxsize

# NOTE: @parallel and @parallel_indices and @parallel_async do not appear in the following as they are extended and therefore re-defined here in parallel.jl
@doc replace(ParallelKernel.SYNCHRONIZE_DOC,        "@init_parallel_kernel" => "@init_parallel_stencil") macro synchronize(args...)        check_initialized(__module__); esc(:(ParallelStencil.ParallelKernel.@synchronize($(args...)))); end


const PARALLEL_DOC = """
    @parallel kernel
    @parallel inbounds=... memopt=... ndims=... kernel
    @parallel inbounds=... memopt=true optvars=... loopdim=... loopsize=... optranges=... optimize_halo_read=... useshmemhalos=... kernel

Declare the `kernel` parallel and containing stencil computations be performed with one of the submodules `ParallelStencil.FiniteDifferences{1D|2D|3D}` (or with a compatible custom module or set of macros).

# Optional keyword arguments
- `inbounds::Bool`: whether to apply `@inbounds` to the kernel. The default is `false` or as set with the `inbounds` keyword argument of [`@init_parallel_stencil`](@ref).
- `memopt::Bool=false`: whether to perform advanced stencil-specific on-chip memory optimisations. Eligible memory-optimized declarations support two or three parallel indices.
- `double_buffering_opt::Bool=true`: whether to attempt to apply the automatic double buffering optimization. When `true` and the kernel signature contains double-buffered (`2B`-suffixed) field/array types (e.g. `Pt::Field2B`, `V::BVectorField2B`), the kernel body is rewritten to avoid race conditions when the kernel has been created by fusion of multiple kernels. After each kernel launch, the `in`/`out` buffers of all `2B` arguments are automatically swapped. When `false` or no `2B` fields are present, this is a no-op. See also the `use_old` keyword arguments below and `swap_double_buffers` keyword arguments of `@parallel` kernel calls (farther below).
- `use_old::Tuple{Vararg{Symbol}}=()`: when set, the listed `2B` fields use the *old* values (`.in`) instead of the freshly-computed on-the-fly values after their on-the-fly definition. This is a numerical optimization that produces a different (cheaper) numerical scheme — the user must explicitly opt in.

!!! note "Advanced optional keyword arguments"
    - `ndims::Integer|Tuple`: the number of dimensions used for the stencil computations in the kernels: 1, 2 or 3 (or a tuple containing any of the previous in order to generate a method for each of the given values - this can only work correctly if the macros used *and loaded* work for any of the chosen values of `ndims`!). A default can be set with the `ndims` keyword argument of [`@init_parallel_stencil`](@ref). The keyword argument `N` becomes mandatory when `ndims` is a tuple in order to dispatch on the number of dimensions (see below).
    - `N::Integer|Tuple`: the value(s) a type parameter `N` in the kernel method signatures must take. The values are typically computed based on `ndims` (set with the corresponding keyword argument of the `@parallel` macro or `@init_parallel_stencil`), which will be substituted in the expression before evaluating it. This enables dispatching on the number of dimensions in the kernel methods (e.g., `@parallel ndims=(1,3) N=ndims function f(A::Data.Array{N}) ... end`). The keyword argument `N` is mandatory if `ndims` is a tuple and must then furthermore be a tuple of the same length as `ndims`.
    - `optvars::Symbol | Tuple{Vararg{Symbol}}`: the read-only stencil input or inputs to optimize explicitly. By default, all eligible read-only multi-point stencil inputs are optimized.
    - `loopdim::Integer`: the optimization dimension used by memory-optimized declarations. Use `2` for 2-D declarations and `3` for 3-D declarations.
    - `loopsize::Integer`: the loop size used by the memory-optimized execution path.
    - `optranges`: per-optimized-array optimization ranges selecting the optimized x-y region and optimization-dimension region.
    - `optimize_halo_read::Bool`: whether halo reads are optimized together with the selected memory-optimization ranges.
    - `useshmemhalos`: per-optimized-array shared-memory halo usage for 3-D memory-optimized declarations.

!!! note "Memory optimization"
    Two-dimensional memory-optimized declarations use register-only caching in the second dimension. Shared-memory caching belongs only to the 3-D memory-optimized declaration surface, where `loopdim=3` and `useshmemhalos` may be used.

See also: [`@init_parallel_stencil`](@ref)

--------------------------------------------------------------------------------
    @parallel kernelcall
    @parallel ∇=... kernelcall

!!! note "Advanced"
        @parallel ranges kernelcall
        @parallel nblocks nthreads kernelcall
        @parallel ranges nblocks nthreads kernelcall
        @parallel (...) configcall=... backendkwargs... kernelcall
        @parallel ∇=... ad_mode=... ad_annotations=... (...) backendkwargs... kernelcall

Declare the `kernelcall` parallel. The kernel will automatically be called as required by the package for parallelization selected with [`@init_parallel_kernel`](@ref) (however, see below the note on automatic computation of `ranges`). Synchronizes at the end of the call (if a stream is given via keyword arguments, then it synchronizes only this stream). If the called kernel was declared with `memopt=true`, this wrapper automatically uses the stored declaration metadata to select the memory-optimized launch-preparation path. The keyword argument `∇` triggers a parallel call to the gradient kernel instead of the kernel itself. The automatic differentiation is performed with the package Enzyme.jl (refer to the corresponding documentation for Enzyme-specific terms used below); Enzyme needs to be imported before ParallelStencil in order to have it load the corresponding extension.

!!! note "Automatic computation of `ranges`"
    Automatic computation of `ranges` for `@parallel <kernelcall>` is only possible if the number of parallel indices used by the kernel is equal to the number of dimensions of the highest-dimensional input arrays. Otherwise, specify the `ranges` manually with `@parallel ranges=... <kernelcall>`.

!!! note "Memory-optimized launches"
    Kernels declared with `memopt=true` activate memory-optimized launch preparation automatically; no repeated launch-time `memopt` keyword is needed. The launch preparation derives ranges, launch parameters, and shared-memory size from the stored declaration metadata when applicable.

!!! note "Runtime hardware selection"
    When KernelAbstractions is chosen as the package for parallelization, this wrapper consults [`current_hardware`](@ref) to determine the runtime hardware target. The symbol defaults to `:cpu` and can be switched to select other targets via [`select_hardware`](@ref).

# Arguments
- `kernelcall`: a call to a kernel that is declared parallel.
!!! note "Advanced optional arguments"
    - `ranges::Tuple{UnitRange{},UnitRange{},UnitRange{}} | Tuple{UnitRange{},UnitRange{}} | Tuple{UnitRange{}} | UnitRange{}`: the ranges of indices in each dimension for which computations must be performed.
    - `nblocks::Tuple{Integer,Integer,Integer}`: the number of blocks to be used if the package CUDA, AMDGPU, Metal or KernelAbstractions was selected with [`@init_parallel_kernel`](@ref).
    - `nthreads::Tuple{Integer,Integer,Integer}`: the number of threads to be used if the package CUDA, AMDGPU, Metal or KernelAbstractions was selected with [`@init_parallel_kernel`](@ref).

# Keyword arguments
!!! note "Advanced"
    - `∇`: the variable(s) with respect to which the kernel is to be differentiated automatically and a duplicate for each variable to store the result in, separated by `->`, e.g., `∇=(A->Ā, B->B̄)`. Setting this keyword triggers a parallel call to the gradient kernel instead of the kernel itself. The duplicate variables are by default passed to Enzyme with the annotation `DuplicatedNoNeed`, e.g., `DuplicatedNoNeed(A, Ā)`. Use the keyword argument `ad_annotations` to modify this behavior.
    - `ad_mode=Enzyme.Reverse`: the automatic differentiation mode (see the documentation of Enzyme.jl for more information).
    - `ad_annotations=()`: Enzyme variable annotations for automatic differentiation in the format `(<keyword>=<variable(s)>, <keyword>=<variable(s)>, ...)`, where `<variable(s)>` can be a single variable or a tuple of variables (e.g., `ad_annotations=(Duplicated=B, Active=(a,b))`). Currently supported annotations are: $(keys(AD_SUPPORTED_ANNOTATIONS)).
    - `configcall=kernelcall`: a call to a kernel that is declared parallel, which is used for determining the kernel launch parameters. This keyword is useful, e.g., for generic automatic differentiation using the low-level submodule [`AD`](@ref).
    - `swap_double_buffers::Bool=true`: whether to automatically swap the `in`/`out` buffers of all double-buffered (`2B`) arguments after the kernel launch. The swap is only performed if the kernel was declared with `2B` fields and `double_buffering_opt=true`; it is suppressed when `launch=false` (compile-only case) or `swap_double_buffers=false`.
    - `backendkwargs...`: keyword arguments to be passed further to CUDA.jl, AMDGPU.jl, Metal.jl or KernelAbstractions.jl (ignored for Threads and Polyester).

!!! note "Performance note"
    Kernel launch parameters are automatically defined with heuristics, where not defined with optional kernel arguments. For CUDA and AMDGPU, `nthreads` is typically set to (32,8,1) and `nblocks` accordingly to ensure that enough threads are launched.

!!! note "Field and data type annotations"
    The host-side `Data.*` / `TData.*` and `Data.Fields.*` / `TData.Fields.*` types (cf. the generated `Data`/`TData` module docstrings) are intended to be usable as argument type annotations in `@parallel` kernel definitions: the parallel kernel declaration and launch system automatically converts these host-side types to the corresponding `Device` types of the active backend so that the resulting device-side kernel signature dispatches correctly when the kernel is launched with a field allocated by the matching field allocation macro (cf. [`ParallelStencil.FieldAllocators`](@ref)).

See also: [`@init_parallel_kernel`](@ref)
"""
@doc PARALLEL_DOC
macro parallel(args...) check_initialized(__module__); checkargs_parallel(args...); esc(parallel(__source__, __module__, args...)); end


##
const PARALLEL_INDICES_DOC = """
    @parallel_indices indices kernel
    @parallel_indices indices inbounds=... memopt=... ndims=... kernel
    @parallel_indices indices inbounds=... memopt=true optvars=... loopdim=... loopsize=... optranges=... optimize_halo_read=... useshmemhalos=... kernel

Declare the `kernel` parallel and generate the given parallel `indices` inside the `kernel` using the package for parallelization selected with [`@init_parallel_stencil`](@ref).

!!! note "Runtime hardware selection"
    When KernelAbstractions is initialized, this wrapper consults [`current_hardware`](@ref) to determine the runtime hardware target. The symbol defaults to `:cpu` and can be switched to select other targets via [`select_hardware`](@ref).

# Optional keyword arguments
    - `inbounds::Bool`: whether to apply `@inbounds` to the kernel. The default is `false` or as set with the `inbounds` keyword argument of [`@init_parallel_stencil`](@ref).
    - `memopt::Bool=false`: whether to perform advanced stencil-specific on-chip memory optimisations. Eligible memory-optimized declarations support two or three generated parallel indices.
    !!! note "Advanced optional keyword arguments"
        - `ndims::Integer|Tuple`: the number of indexing dimensions desired when using splat syntax for the `indices`: 1, 2, 3 (a default `ndims` value can be set with the corresponding keyword argument of [`@init_parallel_stencil`](@ref)) or a tuple containing any of the previous in order to generate a method for each of the given `ndims` values. Concretely, the splat syntax (e.g., `@parallel_indices (I...) ndims=(2,3) ...`) generates a tuple of parallel indices (`I` in this example) where the length is given by the `ndims` value (here `2` for the first method and `3` for the second). This makes it possible to write kernels that are agnostic to the number of dimensions (writing, e.g., `A[I...]` to access elements of the array `A`). The keyword argument `N` becomes mandatory when `ndims` is a tuple in order to dispatch on the number of dimensions (see below).
        - `N::Integer|Tuple`: the value(s) a type parameter `N` in the kernel method signatures must take. The values are typically computed based on `ndims` (set with the corresponding keyword argument of the `@parallel_indices` macro or `@init_parallel_stencil`), which will be substituted in the expression before evaluating it. This enables dispatching on the number of dimensions in the kernel methods (e.g., `@parallel_indices (I...) ndims=(1,3) N=ndims function f(A::Data.Array{N}) ... end`). The keyword argument `N` is mandatory if `ndims` is a tuple and must then furthermore be a tuple of the same length as `ndims`.
        - `optvars::Symbol | Tuple{Vararg{Symbol}}`: the read-only stencil input or inputs to optimize explicitly. By default, all eligible read-only multi-point stencil inputs are optimized.
        - `loopdim::Integer`: the optimization dimension used by memory-optimized declarations. Use `2` for 2-D declarations and `3` for 3-D declarations.
        - `loopsize::Integer`: the loop size used by the memory-optimized execution path.
        - `optranges`: per-optimized-array optimization ranges selecting the optimized x-y region and optimization-dimension region.
        - `optimize_halo_read::Bool`: whether halo reads are optimized together with the selected memory-optimization ranges.
        - `useshmemhalos`: per-optimized-array shared-memory halo usage for 3-D memory-optimized declarations.

!!! note "Memory optimization"
    Two-index memory-optimized declarations use register-only caching in the second dimension. Shared-memory caching belongs only to the 3-D memory-optimized declaration surface, where `loopdim=3` and `useshmemhalos` may be used.

!!! note "Field and data type annotations"
    The host-side `Data.*` / `TData.*` and `Data.Fields.*` / `TData.Fields.*` types (cf. the generated `Data`/`TData` module docstrings) are intended to be usable as argument type annotations in `@parallel_indices` kernel definitions: the parallel kernel declaration and launch system automatically converts these host-side types to the corresponding `Device` types of the active backend so that the resulting device-side kernel signature dispatches correctly when the kernel is launched with a field allocated by the matching field allocation macro (cf. [`ParallelStencil.FieldAllocators`](@ref)).

See also: [`@init_parallel_stencil`](@ref)
"""
@doc PARALLEL_INDICES_DOC
macro parallel_indices(args...) check_initialized(__module__); checkargs_parallel_indices(args...); esc(parallel_indices(__source__, __module__, args...)); end


const PARALLEL_ASYNC_DOC = """
$(replace(ParallelKernel.PARALLEL_ASYNC_DOC, "@init_parallel_kernel" => "@init_parallel_stencil"))
"""
@doc PARALLEL_ASYNC_DOC
macro parallel_async(args...) check_initialized(__module__); checkargs_parallel(args...); esc(parallel_async(__source__, __module__, args...)); end


const ERRMSG_AUTOMATIC_RANGES_PARALLEL = "@parallel <kernelcall>: the ranges needed for the kernel call cannot be automatically computed (less parallel indices than dimensions of the input arrays); specify the ranges manually with @parallel ranges=... <kernelcall>."


## MACROS FORCING PACKAGE, IGNORING INITIALIZATION

macro parallel_cuda(args...)              check_initialized(__module__); checkargs_parallel(args...); esc(parallel(__source__, __module__, args...; package=PKG_CUDA)); end
macro parallel_amdgpu(args...)            check_initialized(__module__); checkargs_parallel(args...); esc(parallel(__source__, __module__, args...; package=PKG_AMDGPU)); end
macro parallel_metal(args...)             check_initialized(__module__); checkargs_parallel(args...); esc(parallel(__source__, __module__, args...; package=PKG_METAL)); end
macro parallel_threads(args...)           check_initialized(__module__); checkargs_parallel(args...); esc(parallel(__source__, __module__, args...; package=PKG_THREADS)); end
macro parallel_polyester(args...)         check_initialized(__module__); checkargs_parallel(args...); esc(parallel(__source__, __module__, args...; package=PKG_POLYESTER)); end
macro parallel_indices_cuda(args...)      check_initialized(__module__); checkargs_parallel_indices(args...); esc(parallel_indices(__source__, __module__, args...; package=PKG_CUDA)); end
macro parallel_indices_amdgpu(args...)    check_initialized(__module__); checkargs_parallel_indices(args...); esc(parallel_indices(__source__, __module__, args...; package=PKG_AMDGPU)); end
macro parallel_indices_metal(args...)     check_initialized(__module__); checkargs_parallel_indices(args...); esc(parallel_indices(__source__, __module__, args...; package=PKG_METAL)); end
macro parallel_indices_threads(args...)   check_initialized(__module__); checkargs_parallel_indices(args...); esc(parallel_indices(__source__, __module__, args...; package=PKG_THREADS)); end
macro parallel_indices_polyester(args...) check_initialized(__module__); checkargs_parallel_indices(args...); esc(parallel_indices(__source__, __module__, args...; package=PKG_POLYESTER)); end
macro parallel_async_cuda(args...)        check_initialized(__module__); checkargs_parallel(args...); esc(parallel_async(__source__, __module__, args...; package=PKG_CUDA)); end
macro parallel_async_amdgpu(args...)      check_initialized(__module__); checkargs_parallel(args...); esc(parallel_async(__source__, __module__, args...; package=PKG_AMDGPU)); end
macro parallel_async_metal(args...)       check_initialized(__module__); checkargs_parallel(args...); esc(parallel_async(__source__, __module__, args...; package=PKG_METAL)); end
macro parallel_async_threads(args...)     check_initialized(__module__); checkargs_parallel(args...); esc(parallel_async(__source__, __module__, args...; package=PKG_THREADS)); end
macro parallel_async_polyester(args...)   check_initialized(__module__); checkargs_parallel(args...); esc(parallel_async(__source__, __module__, args...; package=PKG_POLYESTER)); end


## ARGUMENT CHECKS

function checkargs_parallel(args...)
    posargs, = split_args(args)
    if isempty(posargs) @ArgumentError("arguments missing.") end
    if is_kernel(args[end])  # Case: @parallel kernel
        if (length(posargs) != 1) @ArgumentError("wrong number of (positional) arguments in @parallel kernel call.") end
        kernel = args[end]
        if length(extract_kernel_args(kernel)[2]) > 0 @ArgumentError("keyword arguments are not allowed in the signature of @parallel kernels.") end
    elseif is_call(args[end])  # Case: @parallel <args...> kernelcall
        ParallelKernel.checkargs_parallel(args...)
    else
        @ArgumentError("the last argument must be a kernel definition or a kernel call (obtained: $(args[end])).")
    end
end

function checkargs_parallel_indices(args...)
    posargs, = split_args(args)
    indices = posargs[1]
    if (!isa(indices,Symbol) && !isa(indices.head,Symbol)) @ArgumentError("@parallel_indices: argument 'indices' must be a tuple of indices, a single index or a variable followed by the splat operator representing a tuple of indices (e.g. (ix, iy, iz) or (ix, iy) or ix or I...).") end
    ParallelKernel.checkargs_parallel_indices(posargs...)
end

function check_memopt_supported(memopt::Bool, package::Symbol, context::String)
    if memopt && package == PKG_KERNELABSTRACTIONS
        @KeywordArgumentError("$context: keyword argument `memopt=true` is currently not supported for the KernelAbstractions backend. Set `memopt=false`.")
    end
end

function check_memopt_declaration_args(indices::Union{Symbol,Expr}, loopdim::Integer, useshmemhalos, context::String)
    nb_parallel_indices = isa(indices, Expr) ? length(indices.args) : 1
    if nb_parallel_indices == 2
        if loopdim != 2
            @IncoherentArgumentError("incoherent arguments memopt in $context: two-index kernels require `loopdim=2`.")
        end
        if !isnothing(useshmemhalos)
            @IncoherentArgumentError("incoherent arguments memopt in $context: shared-memory-related keywords are not supported for two-index memory-optimized kernels.")
        end
    elseif nb_parallel_indices == 3
        if loopdim != 3
            @IncoherentArgumentError("incoherent arguments memopt in $context: three-index kernels require `loopdim=3`.")
        end
    else
        @IncoherentArgumentError("incoherent arguments memopt in $context: optimization can only be applied in 2-D and 3-D @parallel kernels and @parallel_indices kernels.")
    end
end


## GATEWAY FUNCTIONS

parallel_async(source::LineNumberNode, caller::Module, args::Union{Symbol,Expr}...; package::Symbol=get_package(caller)) = parallel(source, caller, args...; package=package, async=true)

function parallel(source::LineNumberNode, caller::Module, args::Union{Symbol,Expr}...; package::Symbol=get_package(caller), async::Bool=false)
    if is_kernel(args[end])
        posargs, kwargs_expr, kernelarg = split_parallel_args(args, is_call=false)
        kwargs = extract_kwargs(caller, kwargs_expr, (:ndims, :N, :inbounds, :padding, :memopt, :double_buffering_opt, :use_old, :optvars, :loopdim, :loopsize, :optranges, :useshmemhalos, :optimize_halo_read, :metadata_module, :metadata_function), "@parallel <kernel>"; eval_args=(:ndims, :inbounds, :padding, :memopt, :double_buffering_opt, :loopdim, :optranges, :useshmemhalos, :optimize_halo_read, :metadata_module))
        memopt = haskey(kwargs, :memopt) ? kwargs.memopt : get_memopt(caller)
        check_memopt_supported(memopt, package, "@parallel <kernel>")
        ndims = haskey(kwargs, :ndims) ? kwargs.ndims : get_ndims(caller)
        is_parallel_kernel = true
        if typeof(ndims) <: Tuple
            expand_ndims_tuple(caller, ndims, is_parallel_kernel, kernelarg, kwargs, posargs...)
        else
            if haskey(kwargs, :N)
                substitute_N(caller, ndims, is_parallel_kernel, kernelarg, kwargs, posargs...)
            else
                numbertype = get_numbertype(caller)
                if !haskey(kwargs, :metadata_module)
                    get_name(kernelarg)
                    metadata_module, metadata_function = create_metadata_storage(source, caller, kernelarg)
                else
                    metadata_module, metadata_function = kwargs.metadata_module, kwargs.metadata_function
                end
                parallel_kernel(metadata_module, metadata_function, caller, package, ndims, numbertype, kernelarg, posargs...; kwargs)
            end
        end
    elseif is_call(args[end])
        posargs, kwargs_expr, kernelarg = split_parallel_args(args)
        kwargs, backend_kwargs_expr, ~, kwargs_unknown_dict = extract_kwargs(caller, kwargs_expr, (:memopt, :configcall, :∇, :ad_mode, :ad_annotations, :swap_double_buffers), "@parallel <kernelcall>", true; eval_args=(:memopt, :swap_double_buffers))
        memopt                = haskey(kwargs, :memopt) ? kwargs.memopt : nothing
        if memopt === true check_memopt_supported(true, package, "@parallel <kernelcall>") end
        configcall            = haskey(kwargs, :configcall) ? kwargs.configcall : kernelarg
        configcall_kwarg_expr = :(configcall=$configcall)
        is_ad_highlevel       = haskey(kwargs, :∇)
        if !is_ad_highlevel && (haskey(kwargs, :ad_mode) || haskey(kwargs, :ad_annotations)) @IncoherentArgumentError("incoherent arguments `ad_mode`/`ad_annotations` in @parallel call: AD keywords are only valid if automatic differentiation is triggered with the keyword argument `∇`.") end
        # Read launch and swap_double_buffers for the double-buffering swap gate. `launch` is an unknown kwarg (forwarded to ParallelKernel via backend_kwargs_expr); read it from kwargs_unknown_dict (default true). `swap_double_buffers` is a known kwarg consumed by ParallelStencil (default true).
        launch_val            = haskey(kwargs_unknown_dict, :launch) ? kwargs_unknown_dict[:launch] : true
        swap_double_buffers   = haskey(kwargs, :swap_double_buffers) ? kwargs.swap_double_buffers : true
        if is_ad_highlevel
            # Error out if the kernel was double-buffering-transformed (visible in the metadata via double_buffer_args); automatic differentiation is not yet supported in combination with double buffering.
            metadata_call = create_metadata_call(configcall)
            md_var = gensym("metadata")
            ad_call = ParallelKernel.parallel_call_ad(caller, kernelarg, backend_kwargs_expr, async, package, posargs, kwargs)
            quote
                local $md_var = $metadata_call
                if isdefined($md_var, :double_buffer_args) && !isempty($md_var.double_buffer_args)
                    @ArgumentError("automatic differentiation (∇) is not yet supported in combination with double buffering (the kernel was transformed with the double buffering rewrite pass).")
                end
                $ad_call
            end
        elseif memopt === true
            if (length(posargs) > 1) @ArgumentError("maximum one positional argument (ranges) is allowed in a @parallel memopt=true call.") end
            let
                md_var = gensym("metadata")
                db_swap = build_swap_expr(md_var, configcall.args[2:end], launch_val, swap_double_buffers)
                launch_call = parallel_call_memopt(caller, posargs..., kernelarg, backend_kwargs_expr, async; kwargs...)
                if isnothing(db_swap)
                    launch_call
                else
                    quote
                        local $md_var = $(create_metadata_call(configcall))
                        $launch_call
                        $db_swap
                    end
                end
            end
        elseif memopt === false
            if isempty(posargs)
                ranges = :(ParallelStencil.compute_parallel_ranges(Val(($(create_metadata_call(configcall))).nb_parallel_indices), $(configcall.args[2:end]...)))
                let
                    md_var = gensym("metadata")
                    db_swap = build_swap_expr(md_var, configcall.args[2:end], launch_val, swap_double_buffers)
                    launch_call = ParallelKernel.parallel(caller, ranges, backend_kwargs_expr..., configcall_kwarg_expr, kernelarg; package=package, async=async)
                    if isnothing(db_swap)
                        launch_call
                    else
                        quote
                            local $md_var = $(create_metadata_call(configcall))
                            $launch_call
                            $db_swap
                        end
                    end
                end
            else
                let
                    md_var = gensym("metadata")
                    db_swap = build_swap_expr(md_var, configcall.args[2:end], launch_val, swap_double_buffers)
                    launch_call = ParallelKernel.parallel(caller, posargs..., backend_kwargs_expr..., configcall_kwarg_expr, kernelarg; package=package, async=async)
                    if isnothing(db_swap)
                        launch_call
                    else
                        quote
                            local $md_var = $(create_metadata_call(configcall))
                            $launch_call
                            $db_swap
                        end
                    end
                end
            end
        else
            metadata_call = create_metadata_call(configcall)
            metadata_var = gensym("metadata")
            ordinary_kernelarg = deepcopy(kernelarg)
            ordinary_call = if isempty(posargs)
                ranges = :(ParallelStencil.compute_parallel_ranges(Val($metadata_var.nb_parallel_indices), $(configcall.args[2:end]...)))
                ParallelKernel.parallel(caller, ranges, backend_kwargs_expr..., configcall_kwarg_expr, ordinary_kernelarg; package=package, async=async)
            else
                ParallelKernel.parallel(caller, posargs..., backend_kwargs_expr..., configcall_kwarg_expr, ordinary_kernelarg; package=package, async=async)
            end
            # Build the double-buffering swap expression (emitted after the kernel launch). The swap is gated by launch_val && swap_double_buffers at build time, and by isdefined(metadata, :double_buffer_args) at runtime. When disabled, swap_expr is nothing.
            swap_expr = build_swap_expr(metadata_var, configcall.args[2:end], launch_val, swap_double_buffers)
            if isnothing(swap_expr)
                if isempty(posargs)
                    quote
                        local $metadata_var = $metadata_call
                        if $metadata_var.memopt
                            $(parallel_call_memopt_metadata(caller, metadata_var, kernelarg, backend_kwargs_expr, async; configcall=configcall))
                        else
                            $ordinary_call
                        end
                    end
                elseif length(posargs) == 1
                    quote
                        local $metadata_var = $metadata_call
                        if $metadata_var.memopt
                            $(parallel_call_memopt_metadata(caller, metadata_var, posargs[1], kernelarg, backend_kwargs_expr, async; configcall=configcall))
                        else
                            $ordinary_call
                        end
                    end
                else
                    quote
                        local $metadata_var = $metadata_call
                        if $metadata_var.memopt
                            @ArgumentError("maximum one positional argument (ranges) is allowed in a @parallel memopt=true call.")
                        else
                            $ordinary_call
                        end
                    end
                end
            else
                if isempty(posargs)
                    quote
                        local $metadata_var = $metadata_call
                        if $metadata_var.memopt
                            $(parallel_call_memopt_metadata(caller, metadata_var, kernelarg, backend_kwargs_expr, async; configcall=configcall))
                        else
                            $ordinary_call
                        end
                        $swap_expr
                    end
                elseif length(posargs) == 1
                    quote
                        local $metadata_var = $metadata_call
                        if $metadata_var.memopt
                            $(parallel_call_memopt_metadata(caller, metadata_var, posargs[1], kernelarg, backend_kwargs_expr, async; configcall=configcall))
                        else
                            $ordinary_call
                        end
                        $swap_expr
                    end
                else
                    quote
                        local $metadata_var = $metadata_call
                        if $metadata_var.memopt
                            @ArgumentError("maximum one positional argument (ranges) is allowed in a @parallel memopt=true call.")
                        else
                            $ordinary_call
                        end
                        $swap_expr
                    end
                end
            end
        end
    end
end


function parallel_indices(source::LineNumberNode, caller::Module, args::Union{Symbol,Expr}...; package::Symbol=get_package(caller))
    is_parallel_kernel = false
    numbertype = get_numbertype(caller)
    posargs, kwargs_expr, kernelarg = split_parallel_args(args, is_call=false)
    kwargs = extract_kwargs(caller, kwargs_expr, (:ndims, :N, :inbounds, :padding, :memopt, :double_buffering_opt, :optvars, :loopdim, :loopsize, :optranges, :useshmemhalos, :optimize_halo_read, :metadata_module, :metadata_function), "@parallel_indices"; eval_args=(:ndims, :inbounds, :padding, :memopt, :double_buffering_opt, :loopdim, :optranges, :useshmemhalos, :optimize_halo_read, :metadata_module))
    memopt = haskey(kwargs, :memopt) ? kwargs.memopt : get_memopt(caller)
    check_memopt_supported(memopt, package, "@parallel_indices")
    indices_expr = posargs[1]
    ndims = haskey(kwargs, :ndims) ? kwargs.ndims : get_ndims(caller)
    if typeof(ndims) <: Tuple
        expand_ndims_tuple(caller, ndims, is_parallel_kernel, kernelarg, kwargs, posargs...)
    else
        if haskey(kwargs, :N)
            substitute_N(caller, ndims, is_parallel_kernel, kernelarg, kwargs, posargs...)
        elseif is_splatarg(indices_expr)
            parallel_indices_splatarg(caller, package, ndims, kwargs_expr, posargs..., kernelarg; kwargs)
        else
            if !haskey(kwargs, :metadata_module)
                get_name(kernelarg)
                metadata_module, metadata_function = create_metadata_storage(source, caller, kernelarg)
            else
                metadata_module, metadata_function = kwargs.metadata_module, kwargs.metadata_function
            end
            if !haskey(kwargs, :metadata_module)
                nb_parallel_indices = determine_nb_parallel_indices(caller, get_body(kernelarg), extract_tuple(indices_expr))
                if memopt
                    store_metadata(metadata_module, caller, nb_parallel_indices)
                else
                    store_metadata(metadata_module, caller, nb_parallel_indices; memopt=false)
                end
            end
            inbounds = haskey(kwargs, :inbounds) ? kwargs.inbounds : get_inbounds(caller)
            padding  = haskey(kwargs, :padding)  ? kwargs.padding  : get_padding(caller)
            memopt   = haskey(kwargs, :memopt) ? kwargs.memopt : get_memopt(caller)
            if memopt
                quote
                    $metadata_function
                    $(parallel_indices_memopt(metadata_module, metadata_function, is_parallel_kernel, caller, package, posargs..., kernelarg; kwargs...))  #TODO: the package and numbertype will have to be passed here further once supported as kwargs (currently removed from call: package, numbertype, )
                end
            else
                kwargs_expr = (:(inbounds=$inbounds), :(padding=$padding))
                kernel = ParallelKernel.parallel_indices(caller, posargs..., kwargs_expr..., kernelarg; package=package)
                quote
                    $metadata_function
                    $kernel
                end
            end
        end
    end
end


## @PARALLEL KERNEL FUNCTIONS

function expand_ndims_tuple(caller::Module, ndims::Tuple, is_parallel_kernel::Bool, kernel::Expr, kwargs::NamedTuple, posargs...)
    macroname = (is_parallel_kernel) ? "@parallel" : "@parallel_indices"
    if !(typeof(ndims) <: NTuple{N,<:Integer} where N) @KeywordArgumentError("$macroname: keyword argument 'ndims' must be an integer or a tuple of integers (obtained: $ndims).") end
    if !haskey(kwargs, :N) @KeywordArgumentError("$macroname: keyword argument 'N' is mandatory when 'ndims' is a tuple ('N' must also be present as type parameter in the function signature enabling to dispatch on). ") end
    N = eval_arg(caller, substitute(kwargs.N, :ndims, ndims))
    if !(typeof(N) <: NTuple{length(ndims),<:Integer}) @KeywordArgumentError("$macroname: keyword argument 'N' must be a tuple of integers of the same length as 'ndims' when 'ndims' is a tuple (obtained: N=$N, ndims=$ndims).") end
    kwargs_expr = (:($key=$(getproperty(kwargs, key))) for key in keys(kwargs) if key ∉ (:ndims, :N))
    if (is_parallel_kernel) ndims_methods_expr = (:(@parallel         $(posargs...) ndims=$i N=$n $(kwargs_expr...) $kernel) for (i,n) in zip(ndims,N))
    else                    ndims_methods_expr = (:(@parallel_indices $(posargs...) ndims=$i N=$n $(kwargs_expr...) $kernel) for (i,n) in zip(ndims,N))
    end
    return quote $(ndims_methods_expr...) end
end

function substitute_N(caller::Module, ndims::Integer, is_parallel_kernel::Bool, kernel::Expr, kwargs::NamedTuple, posargs...)
    macroname = (is_parallel_kernel) ? "@parallel" : "@parallel_indices"
    if (ndims < 1 || ndims > 3) @KeywordArgumentError("$macroname: keyword argument 'ndims' is invalid or missing (valid values are 1, 2 or 3; 'ndims' an be set globally in @init_parallel_stencil and overwritten per kernel if needed).") end
    if !haskey(kwargs, :N) @ModuleInternalError("$macroname: substitute_N: function should never be called if keyword argument 'N' is not present.") end
    N = eval_arg(caller, substitute(kwargs.N, :ndims, ndims))
    if !(typeof(N) <: Integer) @KeywordArgumentError("$macroname: keyword argument 'N' must be an integer (or a tuple if 'ndims' is a tuple; obtained: $N).") end
    kwargs_expr = (:($key=$(getproperty(kwargs, key))) for key in keys(kwargs) if key != :N)
    if inexpr_walk(splitdef(kernel)[:whereparams], :N) @IncoherentArgumentError("$macroname: 'N' must not appear in the where clause of the kernel signature when the keyword argument 'N' is used.") end
    kernel = substitute_in_kernel(kernel, :N, N; signature_only=true, typeparams_only=true)
    if (is_parallel_kernel) return :(@parallel         $(posargs...) $(kwargs_expr...) $kernel)
    else                    return :(@parallel_indices $(posargs...) $(kwargs_expr...) $kernel)
    end
end

function parallel_indices_splatarg(caller::Module, package::Symbol, ndims::Integer, kwargs_expr, alias_indices::Expr, kernel::Expr; kwargs::NamedTuple)
    if !@capture(alias_indices, (I_...)) @ArgumentError("@parallel_indices: argument 'indices' must be a tuple of indices, a single index or a variable followed by the splat operator representing a tuple of indices (e.g. (ix, iy, iz) or (ix, iy) or ix or I...).") end
    if (ndims < 1 || ndims > 3) @KeywordArgumentError("@parallel_indices: keyword argument 'ndims' is required for the syntax `@parallel_indices I...` and is invalid or missing (valid values are 1, 2 or 3; 'ndims' an be set globally in @init_parallel_stencil and overwritten per kernel if needed).") end
    indices = get_indices_expr(ndims).args
    indices_expr = Expr(:tuple, indices...)
    kernel = macroexpand(caller, kernel)
    kernel = substitute(kernel, I, indices_expr)
    return :(@parallel_indices $indices_expr $(kwargs_expr...) $kernel)  #TODO: the package and numbertype will have to be passed here further once supported as kwargs (currently removed from signature: package::Symbol, numbertype::DataType, )
end

function parallel_indices_memopt(metadata_module::Module, metadata_function::Expr, is_parallel_kernel::Bool, caller::Module, package::Symbol, indices::Union{Symbol,Expr}, kernel::Expr; ndims::Integer=get_ndims(caller), inbounds::Bool=get_inbounds(caller), padding::Bool=get_padding(caller), memopt::Bool=get_memopt(caller), optvars::Union{Expr,Symbol}=Symbol(""), loopdim::Integer=determine_loopdim(indices), loopsize::Integer=compute_loopsize(package), optranges::Union{Nothing, NamedTuple{t, <:NTuple{N,NTuple{3,UnitRange}} where N} where t}=nothing, useshmemhalos::Union{Nothing, NamedTuple{t, <:NTuple{N,Bool} where N} where t}=nothing, optimize_halo_read::Bool=true)
    if (!memopt) @ModuleInternalError("parallel_indices_memopt: called with `memopt=false` which should never happen.") end
    if (!isa(indices,Symbol) && !isa(indices.head,Symbol)) @ArgumentError("@parallel_indices: argument 'indices' must be a tuple of indices, a single index or a variable followed by the splat operator representing a tuple of indices (e.g. (ix, iy, iz) or (ix, iy) or ix or I...).") end
    if (!isa(optvars,Symbol) && !isa(optvars.head,Symbol)) @KeywordArgumentError("@parallel_indices: keyword argument 'optvars' must be a tuple of optvars or a single optvar (e.g. (A, B, C) or A ).") end
    context = is_parallel_kernel ? "@parallel <kernel>" : "@parallel_indices <kernel>"
    check_memopt_declaration_args(indices, loopdim, useshmemhalos, context)
    body = get_body(kernel)
    body = remove_return(body)
    body = add_memopt(metadata_module, is_parallel_kernel, caller, package, body, indices, optvars, loopdim, loopsize, optranges, useshmemhalos, optimize_halo_read)
    body = add_return(body, package)
    set_body!(kernel, body)
    indices = extract_tuple(indices)
    return :(@parallel_indices $(Expr(:tuple, indices[1:end-1]...)) ndims=$ndims inbounds=$inbounds padding=$padding memopt=false metadata_module=$metadata_module metadata_function=$metadata_function $kernel)  #TODO: the package and numbertype will have to be passed here further once supported as kwargs (currently removed from signature: package::Symbol, numbertype::DataType, )
end

function parallel_kernel(metadata_module::Module, metadata_function::Expr, caller::Module, package::Symbol, ndims::Integer, numbertype::DataType, kernel::Expr; kwargs::NamedTuple)
    is_parallel_kernel = true
    if (ndims < 1 || ndims > 3) @KeywordArgumentError("@parallel: keyword argument 'ndims' is invalid or missing (valid values are 1, 2 or 3; 'ndims' an be set globally in @init_parallel_stencil and overwritten per kernel if needed).") end
    inbounds = haskey(kwargs, :inbounds) ? kwargs.inbounds : get_inbounds(caller)
    padding  = haskey(kwargs, :padding)  ? kwargs.padding  : get_padding(caller)
    memopt = haskey(kwargs, :memopt) ? kwargs.memopt : get_memopt(caller)
    # Read the double-buffering opt (per-kernel kwarg or init default) and the 2B argument positions in the kernel signature; both are stored in the metadata module so the launch wrapper knows which arguments to swap and the AD check can detect double-buffered kernels.
    double_buffering_opt = haskey(kwargs, :double_buffering_opt) ? kwargs.double_buffering_opt : get_double_buffering_opt(caller)
    kernelargs_for_db = splitarg.(extract_kernel_args(kernel)[1])
    double_buffer_args = compute_double_buffer_args(kernelargs_for_db)
    if !haskey(kwargs, :metadata_module)
        if memopt
            store_metadata(metadata_module, caller, ndims; double_buffer_args=double_buffer_args, double_buffering_opt=double_buffering_opt)
        else
            store_metadata(metadata_module, caller, ndims; memopt=false, double_buffer_args=double_buffer_args, double_buffering_opt=double_buffering_opt)
        end
    end
    # Double-buffering body rewrite: when enabled and 2B fields are present, rewrite the body and return a NEW @parallel expression with double_buffering_opt=false injected (to prevent infinite recursion); normal Julia expansion then applies extract_onthefly_arrays!, handle_padding, memopt, etc. to the rewritten body. No-op (returns nothing) otherwise.
    db_result = handle_double_buffering!(metadata_module, metadata_function, caller, package, ndims, numbertype, kernel, nothing; kwargs)
    if !isnothing(db_result)
        return db_result
    end
    # Consume the double-buffering-specific kwargs (double_buffering_opt, use_old) so they do not flow further down the call chain to functions that don't accept them (e.g. parallel_indices_memopt). These kwargs have been consumed by handle_double_buffering! above (and by store_metadata above); they must not be forwarded.
    remaining_keys = filter(k -> k ∉ (:double_buffering_opt, :use_old), keys(kwargs))
    kwargs = NamedTuple{remaining_keys}(kwargs[k] for k in remaining_keys)
    indices = get_indices_expr(ndims).args
    indices_dir = get_indices_dir_expr(ndims).args
    body = get_body(kernel)
    body = remove_return(body)
    validate_body(body)
    kernelargs = splitarg.(extract_kernel_args(kernel)[1])
    argvars = (arg[1] for arg in kernelargs)
    check_mask_macro(caller)
    onthefly_vars, onthefly_exprs, write_vars, body = extract_onthefly_arrays!(body, argvars)
    has_onthefly = !isempty(onthefly_vars)
    body = apply_masks(body, indices)
    body = macroexpand(caller, body)
    body = handle_padding(caller, body, padding, indices; handle_view_accesses=false, delay_dir_handling=has_onthefly && padding) # NOTE: delay_dir_handling is mandatory in case of on-the-fly with padding, because the macros (missing dir_handling) created will only be available in the next world age.
    if has_onthefly
        onthefly_syms  = gensym_world.(onthefly_vars, (@__MODULE__,))
        onthefly_exprs = macroexpand.((caller,), onthefly_exprs)
        onthefly_exprs = handle_padding.((caller,), onthefly_exprs, (padding,), (indices,); handle_view_accesses=false, dir_handling=!padding) # NOTE: dir_handling is done after macro expansion with the delayed handling.
        onthefly_exprs = insert_onthefly!.(onthefly_exprs, (onthefly_vars,), (onthefly_syms,), (indices,), (indices_dir,))
        onthefly_exprs = handle_padding.((caller,), onthefly_exprs, (padding,), (indices,); handle_indexing=false)
        body           = insert_onthefly!(body, onthefly_vars, onthefly_syms, indices, indices_dir)
        create_onthefly_macro.((caller,), onthefly_syms, onthefly_exprs, onthefly_vars, (indices,), (indices_dir,))
    end
    body = handle_padding(caller, body, padding, indices; handle_indexing=false)
    kernel = insert_device_types(caller, kernel) # NOTE: also for CPU, to keep one code path.
    if !memopt
        kernel = adjust_signatures(kernel, package)
        body   = handle_inverses(body)
        body   = handle_indices_and_literals(body, indices, package, numbertype)
        if (inbounds) body = add_inbounds(body) end
    end
    body = add_return(body, package)
    set_body!(kernel, body)
    if memopt
        expanded_kernel = macroexpand(caller, kernel)
        quote
            $metadata_function
            $(parallel_indices_memopt(metadata_module, metadata_function, is_parallel_kernel, caller, package, get_indices_expr(ndims), expanded_kernel; kwargs...)) #TODO: the package and numbertype will have to be passed here further once supported as kwargs (currently removed from call: package, numbertype, )
        end
    else
        if package == PKG_KERNELABSTRACTIONS
            kernel = :(ParallelStencil.ParallelKernel.@ka_kernel $kernel)
        end
        return quote
            $metadata_function
            $kernel
        end # TODO: later could be here called parallel_indices instead of adding the threadids etc above.
    end
end


## @PARALLEL CALL FUNCTIONS

function parallel_call_memopt(caller::Module, metadata_expr::Union{Symbol,Expr}, ranges::Union{Symbol,Expr}, kernelcall::Expr, backend_kwargs_expr::Array, async::Bool; memopt::Bool=false, configcall::Expr=kernelcall)
    if haskey(backend_kwargs_expr, :shmem) @KeywordArgumentError("@parallel <kernelcall>: keyword `shmem` is not allowed when memopt=true is set.") end
    package             = get_package(caller)
    nthreads_x_max      = ParallelKernel.determine_nthreads_x_max(package)
    nthreads_max_memopt = determine_nthreads_max_memopt(package)
    configcall_kwarg_expr = :(configcall=$configcall)
    numbertype      = get_numbertype(caller) # not :(eltype($(optvars)[1])) # TODO: see how to obtain number type properly for each array: the type of the call call arguments corresponding to the optimization variables should be checked
    nblocks_var = gensym("nblocks")
    nthreads_var = gensym("nthreads")
    shmem_var = gensym("shmem")
    range_setup_exprs, range_arg = precompute_parallel_memopt_arg(ranges, "ranges")
    nthreads_nblocks_expr = :(ParallelStencil.compute_memopt_nthreads_nblocks(Val($metadata_expr.loopsizes), Val($metadata_expr.loopdim), Val($metadata_expr.stencilranges), $nthreads_x_max, $nthreads_max_memopt, $range_arg))
    shmem_expr = :(ParallelStencil.compute_memopt_shmem(Val($metadata_expr.shmem_optvars), Val($metadata_expr.use_shmemhalos), Val($metadata_expr.shmem_spans), Val($metadata_expr.shmem_dim1), Val($metadata_expr.shmem_dim2), $nthreads_var, $numbertype))
    if async
        return quote
            $(range_setup_exprs...)
            local $nblocks_var, $nthreads_var = $nthreads_nblocks_expr
            local $shmem_var = $shmem_expr
            @parallel_async memopt=false $configcall_kwarg_expr $range_arg $nblocks_var $nthreads_var shmem=$shmem_var $(backend_kwargs_expr...) $kernelcall
        end
    else
        return quote
            $(range_setup_exprs...)
            local $nblocks_var, $nthreads_var = $nthreads_nblocks_expr
            local $shmem_var = $shmem_expr
            @parallel memopt=false $configcall_kwarg_expr $range_arg $nblocks_var $nthreads_var shmem=$shmem_var $(backend_kwargs_expr...) $kernelcall
        end
    end
end

function parallel_call_memopt_metadata(caller::Module, metadata_expr::Union{Symbol,Expr}, kernelcall::Expr, backend_kwargs_expr::Array, async::Bool; memopt::Bool=false, configcall::Expr=kernelcall)
    package             = get_package(caller)
    nthreads_x_max      = ParallelKernel.determine_nthreads_x_max(package)
    nthreads_max_memopt = determine_nthreads_max_memopt(package)
    ranges_var = gensym("ranges")
    quote
        local $ranges_var = ParallelStencil.compute_memopt_ranges(Val($metadata_expr.is_parallel_kernel), Val($metadata_expr.nb_parallel_indices), Val($metadata_expr.loopdim), $nthreads_x_max, $nthreads_max_memopt, $(configcall.args[2:end]...))
        $(parallel_call_memopt(caller, metadata_expr, ranges_var, kernelcall, backend_kwargs_expr, async; memopt=memopt, configcall=configcall))
    end
end

function parallel_call_memopt_metadata(caller::Module, metadata_expr::Union{Symbol,Expr}, ranges::Union{Symbol,Expr}, kernelcall::Expr, backend_kwargs_expr::Array, async::Bool; memopt::Bool=false, configcall::Expr=kernelcall)
    parallel_call_memopt(caller, metadata_expr, ranges, kernelcall, backend_kwargs_expr, async; memopt=memopt, configcall=configcall)
end

function parallel_call_memopt(caller::Module, ranges::Union{Symbol,Expr}, kernelcall::Expr, backend_kwargs_expr::Array, async::Bool; memopt::Bool=false, configcall::Expr=kernelcall)
    metadata_call = create_metadata_call(configcall)
    metadata_var = gensym("metadata")
    quote
        local $metadata_var = $metadata_call
        $(parallel_call_memopt(caller, metadata_var, ranges, kernelcall, backend_kwargs_expr, async; memopt=memopt, configcall=configcall))
    end
end

function parallel_call_memopt(caller::Module, kernelcall::Expr, backend_kwargs_expr::Array, async::Bool; memopt::Bool=false, configcall::Expr=kernelcall)
    package             = get_package(caller)
    nthreads_x_max      = ParallelKernel.determine_nthreads_x_max(package)
    nthreads_max_memopt = determine_nthreads_max_memopt(package)
    metadata_call       = create_metadata_call(configcall)
    metadata_var = gensym("metadata")
    ranges_var = gensym("ranges")
    quote
        local $metadata_var = $metadata_call
        local $ranges_var = ParallelStencil.compute_memopt_ranges(Val($metadata_var.is_parallel_kernel), Val($metadata_var.nb_parallel_indices), Val($metadata_var.loopdim), $nthreads_x_max, $nthreads_max_memopt, $(configcall.args[2:end]...))
        $(parallel_call_memopt(caller, metadata_var, ranges_var, kernelcall, backend_kwargs_expr, async; memopt=memopt, configcall=configcall))
    end
end


## FUNCTIONS FOR APPLYING OPTIMISATIONS

function add_memopt(metadata_module::Module, is_parallel_kernel::Bool, caller::Module, package::Symbol, body::Expr, indices::Union{Symbol,Expr}, optvars::Union{Expr,Symbol}, loopdim::Integer, loopsize::Integer, optranges::Union{Nothing, NamedTuple{t, <:NTuple{N,NTuple{3,UnitRange}} where N} where t}, useshmemhalos::Union{Nothing, NamedTuple{t, <:NTuple{N,Bool} where N} where t}, optimize_halo_read::Bool)
    memopt(metadata_module, is_parallel_kernel, caller, indices, optvars, loopdim, loopsize, optranges, useshmemhalos, optimize_halo_read, body; package=package)
end


## FUNCTIONS TO DETERMINE OPTIMIZATION PARAMETERS

determine_nthreads_max_memopt(package::Symbol)  = (package == PKG_AMDGPU) ? NTHREADS_MAX_MEMOPT_AMDGPU : ((package == PKG_CUDA) ? NTHREADS_MAX_MEMOPT_CUDA : ((package == PKG_KERNELABSTRACTIONS) ? NTHREADS_MAX_MEMOPT_KERNELABSTRACTIONS : NTHREADS_MAX_MEMOPT_METAL))
determine_loopdim(indices::Union{Symbol,Expr}) = isa(indices,Expr) ? ((length(indices.args)==2) ? 2 : ((length(indices.args)==3) ? 3 : LOOPDIM_NONE)) : LOOPDIM_NONE

function compute_loopsize(package::Symbol)
    compute_capability = get_compute_capability(package)
    if compute_capability == v"∞" # if not set (could also be not CUDA), choose a value that should work well for all architectures, favouring newer ones.
        return 32
    elseif compute_capability < v"8"
        return 16
    elseif compute_capability < v"9"
        return 32
    else
        return 64
    end
end


## FUNCTIONS TO COMPUTE NTHREADS, NBLOCKS, SHARED MEMORY SIZE AND RANGES

function compute_nthreads_memopt(nthreads_x_max, nthreads_max_memopt, maxsize, loopdim, stencilranges) # This is a heuristic, which results typcially in (32,4,1) threads for a 3-D case.
    maxsize = promote_maxsize(maxsize)
    nthreads = ParallelKernel.compute_nthreads(maxsize; nthreads_x_max=nthreads_x_max, nthreads_max=nthreads_max_memopt, flatdim=loopdim)
    for stencilranges_A in values(stencilranges)
        haloextensions = ((length(stencilranges_A[1])-1)*(loopdim!=1), (length(stencilranges_A[2])-1)*(loopdim!=2), (length(stencilranges_A[3])-1)*(loopdim!=3))
        if (2*prod(nthreads) < prod(nthreads .+ haloextensions)) @ArgumentError("@parallel <kernelcall>: the automatic determination of nthreads is not possible for this case. Please specify `nthreads` and `nblocks`.")  end # NOTE: this is a simple heuristic to compute compare the number of threads to the total number of cells including halo.
    end
    # TODO: check if this can simply be removed or even the kernel something need to be adapted:
    if any(maxsize .% nthreads .!= 0) @ArgumentError("@parallel <kernelcall>: memopt optimization not possible for the given maximum array size in the kernel arguments (the maximum array size must be dividable without rest by the number of threads per block)") end # NOTE: this is a requirement for the reading into shared memory because the re-indexing requires that no thread aborts the kernel early. A way around it, is to specify the range such that the condition verified here is true (meaning the range length must be dividable by the number of threads without rest). This can be done automatically for @parallel kernels, because there the array bounds are always verified, but it must be done explicitly by the user for @parallel_indices kernels.
    return nthreads
end

function get_ranges_memopt(nthreads_x_max, nthreads_max_memopt, loopdim, args...)
    ranges   = ParallelKernel.get_ranges(args...)
    maxsize  = length.(ranges)
    nthreads = ParallelKernel.compute_nthreads(maxsize; nthreads_x_max=nthreads_x_max, nthreads_max=nthreads_max_memopt, flatdim=loopdim)
    # TODO: the following code reduces performance from ~482 GB/s to ~478 GB/s
    rests    = maxsize .% nthreads
    ranges_adjustment = ( (rests[1] != 0) ? (nthreads[1] - rests[1]) : 0,
                          (rests[2] != 0) ? (nthreads[2] - rests[2]) : 0,
                          (rests[3] != 0) ? (nthreads[3] - rests[3]) : 0 )
    ranges = ParallelKernel.compute_ranges(maxsize .+ ranges_adjustment) # NOTE: this makes memopt possible also if the maximum array size is not dividable without rest by the number of threads; however, it requires that all array accesses in the kernel are bounds checked. For parallel indices kernels the user has to guarantee that himself.
    return ranges
end


@generated function compute_memopt_shmem(::Val{optvars}, ::Val{use_shmemhalos}, ::Val{shmem_spans}, ::Val{shmem_dim1}, ::Val{shmem_dim2}, nthreads, ::Type{T}) where {optvars, use_shmemhalos, shmem_spans, shmem_dim1, shmem_dim2, T}
    terms = [:(
        (nthreads[$shmem_dim1] + $(getproperty(use_shmemhalos, A)) * $(getproperty(shmem_spans, A)[1])) *
        (nthreads[$shmem_dim2] + $(getproperty(use_shmemhalos, A)) * $(getproperty(shmem_spans, A)[2])) *
        sizeof(T)
    ) for A in optvars]
    if isempty(terms)
        return :(0)
    elseif length(terms) == 1
        return terms[1]
    else
        return Expr(:call, :+, terms...)
    end
end

@generated function compute_memopt_nthreads_nblocks(::Val{loopsizes}, ::Val{loopdim}, ::Val{stencilranges}, nthreads_x_max, nthreads_max_memopt, ranges) where {loopsizes, loopdim, stencilranges}
    return quote
        maxsize = cld.(length.(ParallelStencil.ParallelKernel.promote_ranges(ranges)), $loopsizes)
        nthreads = ParallelStencil.compute_nthreads_memopt(nthreads_x_max, nthreads_max_memopt, maxsize, $loopdim, $stencilranges)
        nblocks = ParallelStencil.ParallelKernel.compute_nblocks(maxsize, nthreads)
        (nblocks, nthreads)
    end
end

@generated function check_nb_parallel_indices(::Val{nb_parallel_indices}, args...) where {nb_parallel_indices}
    errorcall = :(ParallelStencil.@ArgumentError(ParallelStencil.ERRMSG_AUTOMATIC_RANGES_PARALLEL))
    return quote
        nb_input_dims = ParallelStencil.get_nb_input_dims(args...)
        nb_dims_match = (nb_input_dims == $nb_parallel_indices)
        if nb_dims_match isa Bool
            nb_dims_match || $errorcall
        end
        nothing
    end
end

@generated function compute_parallel_ranges(::Val{nb_parallel_indices}, args...) where {nb_parallel_indices}
    return quote
        ParallelStencil.check_nb_parallel_indices(Val($nb_parallel_indices), args...)
        ParallelStencil.ParallelKernel.get_ranges(args...)
    end
end

@generated function compute_memopt_ranges(::Val{is_parallel_kernel}, ::Val{nb_parallel_indices}, ::Val{loopdim}, nthreads_x_max, nthreads_max_memopt, args...) where {is_parallel_kernel, nb_parallel_indices, loopdim}
    if is_parallel_kernel
        range_expr = :(ParallelStencil.get_ranges_memopt(nthreads_x_max, nthreads_max_memopt, $loopdim, args...))
    else
        range_expr = :(ParallelStencil.ParallelKernel.get_ranges(args...))
    end
    errorcall = :(ParallelStencil.@ArgumentError(ParallelStencil.ERRMSG_AUTOMATIC_RANGES_PARALLEL))
    return quote
        nb_input_dims = ParallelStencil.get_nb_input_dims(args...)
        nb_dims_match = (nb_input_dims == $nb_parallel_indices)
        if nb_dims_match isa Bool
            nb_dims_match || $errorcall
        end
        $range_expr
    end
end

function precompute_parallel_memopt_arg(arg::Union{Symbol,Expr}, prefix::AbstractString)
    if isa(arg, Symbol)
        return Expr[], arg
    else
        arg_var = gensym(prefix)
        return [:(local $arg_var = $arg)], arg_var
    end
end


## FUNCTIONS TO DEAL WITH MASKS (@WITHIN) AND INDICES

is_splatarg(x) = isa(x,Expr) && (x.head == :...)

function check_mask_macro(caller::Module)
    if !isdefined(caller, Symbol("@within")) @MethodPluginError("the macro @within is not defined in the caller. You need to load one of the submodules ParallelStencil.FiniteDifferences{1|2|3}D (or a compatible custom module or set of macros).") end
    methods_str = string(methods(getfield(caller, Symbol("@within"))))
    if !occursin(r"(var\"@within\"|@within)\(__source__::LineNumberNode, __module__::Module, .*::String, .*\)", methods_str) @MethodPluginError("the signature of the macro @within is not compatible with ParallelStencil (detected signature: \"$methods_str\"). The signature must correspond to the description in ParallelStencil.WITHIN_DOC. See in ParallelStencil.FiniteDifferences{1|2|3}D for examples.") end
end

function apply_masks(expr::Expr, indices::Array{Any}; do_shortif=false)
    args = expr.args
    for i=1:length(args)
        if typeof(args[i]) == Expr
            e = args[i]
            if e.head == :(=) && typeof(e.args[1]) == Expr && e.args[1].head == :macrocall
                lefthand_macro = e.args[1].args[1]
                lefthand_var   = e.args[1].args[3]
                macroname = string(lefthand_macro)
                if do_shortif
                    args[i] = quote
                        $(e.args[1]) = (@within($macroname, $lefthand_var)) ? $(e.args[2]) : $(e.args[1]) #TODO: as else-variable (currently e.args[1], e.g. T2), the right value from the righthand side should be taken (e.g. T) for best perf. This requires though user indication of the kind T2==T as kwarg... Also, these shortifs can only be used if the within conditions are centered (1 < ... < size(A,1)-1) instead of as present ((0 < ... < size(A,1)-2)).
                    end
                else
                    args[i] = quote
                                if (@within($macroname, $lefthand_var))
                                    $e
                                end
                            end
                end
            else
                args[i] = apply_masks(e, indices; do_shortif=do_shortif)
            end
        end
    end
    return expr
end

function get_indices_expr(ndims::Integer)
    if ndims == 1
        return :($(INDICES[1]),)
    elseif ndims == 2
        return :($(INDICES[1]), $(INDICES[2]))
    elseif ndims == 3
        return :($(INDICES[1]), $(INDICES[2]), $(INDICES[3]))
    else
        @ModuleInternalError("argument 'ndims' must be 1, 2 or 3.")
    end
end

function get_indices_dir_expr(ndims::Integer)
    if ndims == 1
        return :($(INDICES_DIR[1]),)
    elseif ndims == 2
        return :($(INDICES_DIR[1]), $(INDICES_DIR[2]))
    elseif ndims == 3
        return :($(INDICES_DIR[1]), $(INDICES_DIR[2]), $(INDICES_DIR[3]))
    else
        @ModuleInternalError("argument 'ndims' must be 1, 2 or 3.")
    end
end

function determine_nb_parallel_indices(caller::Module, body::Expr, indices)
    body = macroexpand(caller, body)
    used_indices = filter(index -> inexpr_walk(body, index), indices)
    if 0 < length(used_indices) < length(indices)
        unused_indices = filter(index -> !inexpr_walk(body, index), indices)
        @ArgumentError("@parallel_indices: all parallel indices must be used in the kernel body (unused indices: $(join(string.(unused_indices), ", "))).")
    end
    return length(indices)
end


## FUNCTIONS TO CREATE METADATA STORAGE

function create_metadata_storage(source::LineNumberNode, caller::Module, kernel::Expr)
    kernelid = get_kernelid(kernel, source.file, source.line)
    create_module(caller, MOD_METADATA_PS)
    topmodule = @eval(caller, $MOD_METADATA_PS)
    create_module(topmodule, kernelid)
    metadata_module = @eval(topmodule, $kernelid)
    metadata_function = create_metadata_function(kernel, metadata_module)
    return metadata_module, metadata_function
end

function create_module(hostmodule::Module, modulename::Symbol; do_baremodule=true)
    if !isdefined(hostmodule, modulename)
        moduleexpr = (do_baremodule) ? :(baremodule $modulename end) : :(module $modulename end)
        @eval(hostmodule, $moduleexpr)
    end
end

function create_metadata_function(kernel::Expr, metadata_module::Module) # NOTE: unlike the creation of the module above, the creation of the matter data function has to happen every time: if we redefine the same function we to have to redefine the meta data...
    metadata_function = deepcopy(kernel)
    kernelname = get_name(kernel)
    functionname = get_meta_function(kernelname)
    metadata_function = set_name(metadata_function, functionname)
    set_body!(metadata_function, quote
        return $metadata_module
    end)
    return :(@inline $metadata_function)
end

function create_metadata_call(configcall::Expr)
    metadata_call = deepcopy(configcall)
    kernelname = metadata_call.args[1]
    metadata_call.args[1] = get_meta_function(kernelname)
    return metadata_call
end

function store_metadata(metadata_module::Module, caller::Module, nb_parallel_indices::Integer; memopt::Union{Nothing,Bool}=nothing, double_buffer_args::Union{Nothing,Tuple{Vararg{Int}}}=nothing, double_buffering_opt::Union{Nothing,Bool}=nothing)
    nonconst_metadata = get_nonconst_metadata(caller)
    if nonconst_metadata || isdefined(metadata_module, :nb_parallel_indices)
        if isnothing(memopt)
            storeexpr = quote
                nb_parallel_indices = $nb_parallel_indices
            end
        else
            storeexpr = quote
                nb_parallel_indices = $nb_parallel_indices
                memopt = $memopt
            end
        end
    else
        if isnothing(memopt)
            storeexpr = quote
                const nb_parallel_indices = $nb_parallel_indices
            end
        else
            storeexpr = quote
                const nb_parallel_indices = $nb_parallel_indices
                const memopt = $memopt
            end
        end
    end
    @eval(metadata_module, $storeexpr)
    # Store double_buffering_opt and double_buffer_args (the 2B argument positions) so the launch wrapper knows whether the kernel was double-buffering-transformed and which arguments to swap after launch. Both honor nonconst_metadata (first definition is const); double_buffer_args is only stored when non-empty.
    db_storeexprs = Expr(:block)
    if !isnothing(double_buffering_opt)
        if nonconst_metadata || isdefined(metadata_module, :double_buffering_opt)
            push!(db_storeexprs.args, :(double_buffering_opt = $double_buffering_opt))
        else
            push!(db_storeexprs.args, :(const double_buffering_opt = $double_buffering_opt))
        end
    end
    if !isnothing(double_buffer_args) && !isempty(double_buffer_args)
        if nonconst_metadata || isdefined(metadata_module, :double_buffer_args)
            push!(db_storeexprs.args, :(double_buffer_args = $double_buffer_args))
        else
            push!(db_storeexprs.args, :(const double_buffer_args = $double_buffer_args))
        end
    end
    if !isempty(db_storeexprs.args)
        @eval(metadata_module, $db_storeexprs)
    end
end

get_kernelid(kernelname, file, line) = Symbol("$(kernelname)_$(file)_$(line)")
get_kernelid(kernel::Expr, file, line) = Symbol("$(get_kernelid(get_name(kernel), file, line))_$(hash(string(kernel)))")
get_meta_function(kernelname)        = Symbol("$(META_FUNCTION_PREFIX)$(GENSYM_SEPARATOR)$(kernelname)")


## FUNCTIONS TO DEAL WITH ON-THE-FLY ASSIGNMENTS

function extract_onthefly_arrays!(body, argvars)
    onthefly_vars  = ()
    onthefly_exprs = ()
    write_vars     = ()
    statements     = get_statements(body)
    for statement in statements
        if is_array_assignment(statement)
            if !@capture(statement, @m_(A_) = assign_expr_) @ArgumentError(ERRMSG_KERNEL_UNSUPPORTED) end
            if any(inexpr_walk.((A,), argvars))
                write_vars = (write_vars..., A)
            end
        end
    end
    for statement in statements
        if is_array_assignment(statement)
            if !@capture(statement, @m_(A_) = assign_expr_) @ArgumentError(ERRMSG_KERNEL_UNSUPPORTED) end
            if !any(inexpr_walk.((A,), argvars))
                if (m != Symbol("@all"))         @ArgumentError("unsupported kernel statements in @parallel kernel definition: partial assignments are not possible for arrays that are not stored in global memory (arrays that are not among the arguments of the kernel); use '@all' instead.") end
                if (inexpr_walk(assign_expr, A)) @ArgumentError("unsupported kernel statements in @parallel kernel definition: auto-dependency is not possible for arrays that are not stored in global memory (arrays that are not among the arguments of the kernel).") end
                if any(inexpr_walk.((assign_expr,), write_vars)) # NOTE: in this case here could later be allocated a local array instead
                    @ArgumentError("unsupported kernel statements in @parallel kernel definition: the assignment of $A should be done on the fly as it is not among the arguments of the kernel; however, this is not possible because it depends on at least one variable that is not read-only within the scope of the kernel (any of: $write_vars).")
                else
                    onthefly_vars  = (onthefly_vars..., A)
                    onthefly_exprs = (onthefly_exprs..., assign_expr)
                    body           = substitute(body, statement, NOEXPR)
                end
            end
        end
    end
    return onthefly_vars, onthefly_exprs, write_vars, body
end

function insert_onthefly!(expr, onthefly_vars, onthefly_syms, indices::Array, indices_dir::Array)
    indices = (indices...,)
    indices_dir = (indices_dir...,)
    for (A, m) in zip(onthefly_vars, onthefly_syms)
        expr = substitute(expr, A, m, indices, indices_dir)
    end
    return expr
end

function determine_local_index_dir(local_index, dim)
    id_l = local_index
    id_l = increment_arg(id_l, INDICES_DIR_FUNCTIONS_SYMS[dim])
    id_l = substitute(id_l, INDICES_DIR[dim], :($(INDICES_DIR_FUNCTIONS_SYMS[dim])(2)))
    id_l = substitute(id_l, INDICES[dim], INDICES_DIR[dim])
    return id_l
end

function create_onthefly_macro(caller, m, expr, var, indices, indices_dir)
    ndims                 = length(indices)
    ix, iy, iz            = gensym_world.(("ix","iy","iz"), (@__MODULE__,))
    ixd, iyd, izd         = gensym_world.(("ixd","iyd","izd"), (@__MODULE__,))
    local_indices         = (ndims==3) ? (ix, iy, iz) : (ndims==2) ? (ix, iy) : (ix,)
    local_indices_dir     = (ndims==3) ? (ixd, iyd, izd) : (ndims==2) ? (ixd, iyd) : (ixd,)
    for (index, local_index) in zip(indices, local_indices)
        expr = substitute(expr, index, Expr(:$, local_index))
    end
    for (index, local_index) in zip(indices_dir, local_indices_dir)
        expr = substitute(expr, index, Expr(:$, local_index))
    end
    local_assign = quote
        $((:($(local_indices_dir[i]) = ParallelStencil.determine_local_index_dir($(local_indices[i]), $i)) for i=1:ndims)...)
    end
    expr_quoted = :($(Expr(:quote, expr)))
    m_function = :($m($(local_indices...)) = ($local_assign; $expr_quoted))
    m_macro = :(macro $m(args...) if (length(args)!=$ndims) ParallelStencil.@ArgumentError("unsupported kernel statements in @parallel kernel definition: wrong number of indices in $var (expected $ndims indices).") end; esc($m(args...)) end)
    @eval(caller, $m_function)
    @eval(caller, $m_macro)
    return
end


## FUNCTIONS TO DEAL WITH DOUBLE BUFFERING

# Extract the leaf type name from a type annotation expression. The annotation can be: a bare Symbol (e.g. :Field2B, :Field), a qualified `.` expression (e.g. :(Data.Fields.Field2B), :(Data.Number)), or a parameterized `curly` expression (e.g. :(Data.Fields.BVectorField2B{3, (:x,:y,:z)})). Returns the leaf Symbol (e.g. :Field2B, :Number), or `nothing` if it cannot be extracted.
function extract_typename(type_expr::Symbol)
    return type_expr
end

function extract_typename(type_expr::Expr)
    if type_expr.head == :.
        lastarg = type_expr.args[end]
        return (lastarg isa QuoteNode) ? lastarg.value : (lastarg isa Symbol) ? lastarg : nothing
    elseif type_expr.head == :curly
        return extract_typename(type_expr.args[1])
    elseif type_expr.head == :(::)
        # e.g. :(Pt::Field2B) — but splitarg already separates name/type, so this shouldn't occur; handle defensively
        return extract_typename(type_expr.args[end])
    else
        return nothing
    end
end

extract_typename(::Any) = nothing

# Check whether a type annotation denotes a double-buffered ("2B") type. A type is 2B if its leaf name ends with the suffix "2B" (e.g. Field2B, BVectorField2B, Array2B, SubArray2B, XField2B, ...). This is purely syntactic (no semantic interpretation of parameters) and matches the FIELDTYPES/ARRAYTYPES entries added for the 2B feature.
function is_2B_type(type_expr)
    name = extract_typename(type_expr)
    return !isnothing(name) && endswith(string(name), "2B")
end

# Compute the positions (1-based) of double-buffered ("2B") arguments in a kernel signature. `kernelargs` is the result of `splitarg.(extract_kernel_args(kernel)[1])`, where each element is a tuple `(name, type, slack, default)` from MacroTools.splitarg. Returns a tuple of Int positions (e.g. (1, 2) if the 1st and 2nd args are 2B).
function compute_double_buffer_args(kernelargs)
    positions = Int[]
    for (i, ka) in enumerate(kernelargs)
        type_expr = ka[2]
        if !isnothing(type_expr) && is_2B_type(type_expr)
            push!(positions, i)
        end
    end
    return (positions...,)
end

# Substitute all occurrences of a bare Symbol `A` in `expr` with the replacement `new`. The replacement can be a Symbol (e.g. A_onthefly) or an Expr (e.g. :(A.in)). Only replaces bare Symbol occurrences (not field accesses like A.in which are Expr).
function substitute_symbol(expr::Symbol, A::Symbol, new)
    return (expr == A) ? new : expr
end

function substitute_symbol(expr::Expr, A::Symbol, new)
    return postwalk(expr) do ex
        (ex isa Symbol && ex == A) ? new : ex
    end
end

substitute_symbol(expr, A::Symbol, new) = expr

# Build the runtime swap expression for double-buffered arguments, emitted after the kernel launch and gated by `launch_val && swap_double_buffers && isdefined(metadata, :double_buffer_args)`; for each position in `metadata_var.double_buffer_args` the argument is swapped: `arg = (in=arg.out, out=arg.in)`.
function build_swap_expr(metadata_var::Symbol, args::Vector{Any}, launch_val::Bool, swap_double_buffers::Bool)
    if !launch_val || !swap_double_buffers
        return nothing  # swap disabled — no expression to emit
    end
    # Generate conditional swaps for each argument position (1-based). Each swap is a direct `if` check (no `let` scope) to avoid soft-scope issues: the swap assignments (e.g. `Pt = (in=Pt.out, out=Pt.in)`) must modify the caller's variables, whether they are global or local. Using a `let` block or `global` keyword would break in local-scope contexts (functions, @testset).
    conditional_swaps = Expr[]
    for (i, arg) in enumerate(args)
        if arg isa Symbol
            swap = :($arg = (in = $(arg).out, out = $(arg).in))
            check = :(isdefined($metadata_var, :double_buffer_args) && !isempty($metadata_var.double_buffer_args) && $i in $metadata_var.double_buffer_args)
            push!(conditional_swaps, :(if $check; $swap; end))
        end
    end
    if isempty(conditional_swaps)
        return nothing
    end
    return quote $(conditional_swaps...) end
end

# Check if a statement is an array assignment with LHS @all(A) for a given A. @all(A) is a macrocall: Expr(:macrocall, Symbol("@all"), LineNumberNode, A).
function is_all_assignment_to(statement, A::Symbol)
    if !is_array_assignment(statement) return false end
    lhs = statement.args[1]  # the macrocall @m_(...)
    if lhs.head != :macrocall return false end
    if lhs.args[1] != Symbol("@all") return false end
    # @all(A) has args[3] = A (a Symbol); @all(A.x) has args[3] = :(A.x)
    target = lhs.args[3]
    if target isa Symbol return target == A end
    if target isa Expr && target.head == :. return target.args[1] == A end
    return false
end

# Extract the field A from a statement's LHS macrocall (e.g. @all(A) → A, @inn(A.x) → A). Returns the Symbol of the field being written to, or nothing if not an array assignment.
function get_lhs_field(statement)
    if !is_array_assignment(statement) return nothing end
    lhs = statement.args[1]
    if lhs.head != :macrocall return nothing end
    target = lhs.args[3]
    if target isa Symbol return target
    elseif target isa Expr && target.head == :. return target.args[1]
    elseif target isa Expr && target.head == :ref return target.args[1]
    end
    return nothing
end

# Check if a statement's LHS uses @all (vs @inn/@d_xi/etc which are partial updates).
function is_all_assignment(statement)
    if !is_array_assignment(statement) return false end
    lhs = statement.args[1]
    return lhs.head == :macrocall && lhs.args[1] == Symbol("@all")
end

# Rewrite the LHS of a statement: replace the bare field A with the replacement expr. E.g. @inn(A.x) with A→A.out becomes @inn(A.out.x).
function rewrite_lhs_field(statement, A::Symbol, new)
    stmt = deepcopy(statement)
    lhs = stmt.args[1]  # macrocall
    lhs.args[3] = substitute_symbol(lhs.args[3], A, new)
    return stmt
end

# The main double-buffering rewrite pass. When double_buffering_opt=true and the kernel signature contains 2B fields, this rewrites the body (steps 0-3 from the spec) and returns a NEW @parallel expression with double_buffering_opt=false injected (to prevent infinite recursion), use_old consumed, and all other kwargs preserved. The returned expression is returned all the way back to the user code and let normal Julia expansion handle it (no macroexpand call on it). When no 2B fields are present or the opt is false, returns the kernel unchanged (no-op early-return).
function handle_double_buffering!(metadata_module::Module, metadata_function::Expr, caller::Module, package::Symbol, ndims::Integer, numbertype::DataType, kernel::Expr, posargs; kwargs::NamedTuple)
    double_buffering_opt = haskey(kwargs, :double_buffering_opt) ? kwargs.double_buffering_opt : get_double_buffering_opt(caller)
    # Early no-op return when the optimization is disabled.
    if !double_buffering_opt
        return nothing
    end
    # Compute 2B arg positions and names.
    kernelargs = splitarg.(extract_kernel_args(kernel)[1])
    double_buffer_args = compute_double_buffer_args(kernelargs)
    # Early no-op return when no 2B fields are present.
    if isempty(double_buffer_args)
        return nothing
    end
    # Collect the 2B field names (Symbols) from the signature.
    db_fields = Symbol[ kernelargs[i][1] for i in double_buffer_args ]
    # Read the use_old option (a tuple of field names to use old values for, or nothing).
    use_old = haskey(kwargs, :use_old) ? kwargs.use_old : ()
    # use_old may arrive as an unevaluated Expr (e.g. :(Pt,) or :((Pt, V))) since it is not in eval_args (the field names are Symbols, not variables to evaluate). Extract the Symbols; if it's already a Tuple of Symbols, use it directly.
    if use_old isa Expr && use_old.head == :tuple
        use_old = Tuple(arg isa QuoteNode ? arg.value : arg for arg in use_old.args)
    elseif use_old isa Symbol
        use_old = (use_old,)
    elseif !isa(use_old, Tuple)
        use_old = ()
    end
    use_old_set = Set(use_old)

    body = get_body(kernel)
    # NOTE: do NOT call remove_return here — the downstream @parallel expansion will handle the return statement. We only need to read the body statements.
    statements = get_statements(body)

    # Classify 2B fields: @all-updated vs partial-updated. Also check that no field has more than one @all(A)=... assignment (not allowed: the user must merge multiple updates into a single @all(A)=... statement; for on-the-fly variables, use a different variable for each case).
    all_updated = Symbol[]
    partial_updated = Symbol[]
    for A in db_fields
        all_count = 0
        has_partial = false
        for stmt in statements
            if is_array_assignment(stmt)
                fld = get_lhs_field(stmt)
                if fld == A
                    if is_all_assignment(stmt)
                        all_count += 1
                    else
                        has_partial = true
                    end
                end
            end
        end
        if all_count > 1
            @ArgumentError("unsupported kernel statements in @parallel kernel definition: multiple @all($A) = ... statements for the same field $A are not allowed in a kernel; merge them into a single @all($A) = ... (or use different variable names for on-the-fly variables).")
        end
        if all_count > 0
            push!(all_updated, A)
        elseif has_partial
            push!(partial_updated, A)
        end
    end

    # Step 0: For partial-updated 2B fields, check A is not used after the update; rewrite LHS to A.out.
    for A in partial_updated
        first_update_idx = 0
        for (i, stmt) in enumerate(statements)
            if is_array_assignment(stmt) && get_lhs_field(stmt) == A && !is_all_assignment(stmt)
                first_update_idx = i
                break
            end
        end
        # Check A is not used after the update (on the RHS of any subsequent statement).
        for i in (first_update_idx+1):length(statements)
            stmt = statements[i]
            if isa(stmt, Expr) && inexpr_walk(stmt, A) && !(is_array_assignment(stmt) && get_lhs_field(stmt) == A)
                @ArgumentError("unsupported kernel statements in @parallel kernel definition: the double-buffered field $A is used after its partial update, which is not allowed (it would read a mix of old and new values).")
            end
        end
    end

    # Build the new statements list (step 0 LHS rewrite for partial-updated fields, step 1 on-the-fly creation for @all-updated fields, steps 2-3 RHS rewriting).
    new_statements = Any[]
    # Track for each @all-updated field: its onthefly symbol, and whether we've passed its def.
    onthefly_syms = Dict{Symbol,Symbol}()
    onthefly_seen = Set{Symbol}()  # fields whose onthefly def has been emitted
    for A in all_updated
        onthefly_syms[A] = gensym_world("$(A)_onthefly", caller)
    end
    # Helper: build the :(A.in) or :(A.out) expression for a given field Symbol A.
    make_in(sym::Symbol) = Expr(:., sym, QuoteNode(:in))
    make_out(sym::Symbol) = Expr(:., sym, QuoteNode(:out))

    for stmt in statements
        if !is_array_assignment(stmt)
            push!(new_statements, stmt)
            continue
        end
        fld = get_lhs_field(stmt)

        if fld in all_updated && is_all_assignment(stmt)
            # Step 1: @all(A) = RHS  →  @all(A_onthefly) = RHS[A→A.in]
            A = fld
            onthefly_sym = onthefly_syms[A]
            new_rhs = substitute_symbol(stmt.args[2], A, make_in(A))
            # Rewrite @all(A) → @all(A_onthefly) on LHS
            new_lhs = deepcopy(stmt.args[1])
            new_lhs.args[3] = onthefly_sym
            push!(new_statements, Expr(:(=), new_lhs, new_rhs))
            # Step 2: insert @all(A.out) = @all(A_onthefly) after
            out_lhs = deepcopy(stmt.args[1])
            out_lhs.args[3] = make_out(A)
            out_rhs_lhs = deepcopy(stmt.args[1])  # @all(...)
            out_rhs_lhs.args[3] = onthefly_sym
            push!(new_statements, Expr(:(=), out_lhs, out_rhs_lhs))
            push!(onthefly_seen, A)
        elseif fld in partial_updated
            # Step 0: rewrite LHS @inn(A[...]) → @inn(A.out[...])
            A = fld
            new_stmt = rewrite_lhs_field(stmt, A, make_out(A))
            # Step 3b: rewrite RHS A → A.in (for the partial-updated field itself)
            new_rhs = substitute_symbol(deepcopy(stmt.args[2]), A, make_in(A))
            # Also rewrite RHS for @all-updated 2B fields (step 3a)
            for B in all_updated
                if B in onthefly_seen && !(B in use_old_set)
                    new_rhs = substitute_symbol(new_rhs, B, onthefly_syms[B])
                else
                    new_rhs = substitute_symbol(new_rhs, B, make_in(B))
                end
            end
            new_stmt.args[2] = new_rhs
            push!(new_statements, new_stmt)
        else
            # Non-2B field assignment: rewrite RHS for any 2B fields appearing on RHS.
            new_rhs = deepcopy(stmt.args[2])
            for A in all_updated
                if A in onthefly_seen && !(A in use_old_set)
                    # Step 3a (non-use_old): after the onthefly def, A → A_onthefly
                    new_rhs = substitute_symbol(new_rhs, A, onthefly_syms[A])
                else
                    # use_old, or before the onthefly def: A → A.in
                    new_rhs = substitute_symbol(new_rhs, A, make_in(A))
                end
            end
            for A in partial_updated
                # Step 3b: A → A.in
                new_rhs = substitute_symbol(new_rhs, A, make_in(A))
            end
            new_stmt = deepcopy(stmt)
            new_stmt.args[2] = new_rhs
            push!(new_statements, new_stmt)
        end
    end

    # Rebuild the kernel with the new body.
    new_kernel = deepcopy(kernel)
    set_body!(new_kernel, Expr(:block, new_statements...))

    # Build the kwargs for the returned @parallel expression: inject double_buffering_opt=false, consume use_old, preserve all other kwargs.
    kwargs_expr = Expr[]
    for key in keys(kwargs)
        if key == :double_buffering_opt
            push!(kwargs_expr, :(double_buffering_opt = false))
        elseif key == :use_old
            # consumed — do not forward
        else
            push!(kwargs_expr, Expr(:(=), key, getproperty(kwargs, key)))
        end
    end
    # If double_buffering_opt was not in kwargs (it came from the init default), inject it now.
    if !(:double_buffering_opt in keys(kwargs))
        push!(kwargs_expr, :(double_buffering_opt = false))
    end

    return :(ParallelStencil.@parallel $(kwargs_expr...) $new_kernel)
end


## FUNCTIONS TO CHECK THE AUTOMATIC DETERMINATION OF RANGES AND NB_PARALLEL_INDICES

function add_nb_parallel_indices_check(ranges::Union{Symbol,Expr}, configcall::Expr)
    metadata_call       = create_metadata_call(configcall)
    return :(ParallelStencil.check_nb_parallel_indices(Val(($metadata_call).nb_parallel_indices), $(configcall.args[2:end]...)); $ranges)
end

get_nb_input_dims(args...)                             = maximum((get_nb_input_dims(arg) for arg in args); init=1)
get_nb_input_dims(t::T) where T<:Union{Tuple,NamedTuple} = get_nb_input_dims(t...)
get_nb_input_dims(A::AbstractArray)                   = ndims(A)
get_nb_input_dims(A::SubArray)                        = ndims(A.parent)
get_nb_input_dims(a::Number)                          = 1
get_nb_input_dims(x)                                  = isbitstype(typeof(x)) ? 1 : @ArgumentError("automatic detection of ranges not possible in @parallel <kernelcall>: some kernel arguments are neither arrays nor scalars nor any other bitstypes nor (named) tuple containing any of the former. Specify ranges or nthreads and nblocks manually.")