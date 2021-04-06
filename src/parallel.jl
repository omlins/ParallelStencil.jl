import .ParallelKernel: get_body, set_body!, add_return, remove_return, substitute, literaltypes, push_to_signature!, add_loop, add_threadids, RANGES_VARNAME, RANGES_TYPE, RANGELENGTHS_VARNAMES

const PARALLEL_DOC = """
    @parallel kernel

Declare the `kernel` parallel and containing stencil computations be performed with one of the submodules `ParallelStencil.FiniteDifferences{1D|2D|3D}` (or with a compatible custom module or set of macros).

See also: [`@init_parallel_stencil`](@ref)

--------------------------------------------------------------------------------
$(replace(ParallelKernel.PARALLEL_DOC, "@init_parallel_kernel" => "@init_parallel_stencil"))
"""
@doc PARALLEL_DOC
macro parallel(args...) check_initialized(); checkargs_parallel(args...); esc(parallel(__module__, args...)); end


## MACROS FORCING PACKAGE, IGNORING INITIALIZATION

macro parallel_cuda(args...)    check_initialized(); checkargs_parallel(args...); esc(parallel(__module__, args...; package=PKG_CUDA)); end
macro parallel_threads(args...) check_initialized(); checkargs_parallel(args...); esc(parallel(__module__, args...; package=PKG_THREADS)); end


## ARGUMENT CHECKS

function checkargs_parallel(args...)
    if isempty(args) @ArgumentError("arguments missing.") end
    if is_function(args[end])  # Case: @parallel kernel
        if (length(args) != 1) @ArgumentError("wrong number of arguments in @parallel kernel call.") end
    elseif is_call(args[end])  # Case: @parallel <args...> kernelcall
        ParallelKernel.checkargs_parallel(args...)
    else
        @ArgumentError("the last argument must be a function definition or a kernel call (obtained: $(args[end])).")
    end
end


## GATEWAY FUNCTIONS

function parallel(caller::Module, args::Union{Symbol,Expr}...; package::Symbol=get_package())
    if is_function(args[end])
        numbertype = get_numbertype()
        ndims      = get_ndims()
        parallel_kernel(caller, package, numbertype, ndims, args...)
    elseif is_call(args[end])
        ParallelKernel.parallel(args...)
    end
end


## @PARALLEL KERNEL FUNCTIONS

function parallel_kernel(caller::Module, package::Symbol, numbertype::DataType, ndims::Integer, kernel::Expr)
    indices = get_indices_expr(ndims).args
    body = get_body(kernel)
    body = remove_return(body)
    validate_body(body)
    if (package == PKG_CUDA)
        kernel = substitute(kernel, :(Data.Array), :(Data.DeviceArray))
    end
    kernel = push_to_signature!(kernel, :($RANGES_VARNAME::$RANGES_TYPE))
    if (package == PKG_CUDA)
        kernel = push_to_signature!(kernel, :($(RANGELENGTHS_VARNAMES[1])::$INT_CUDA))
        kernel = push_to_signature!(kernel, :($(RANGELENGTHS_VARNAMES[2])::$INT_CUDA))
        kernel = push_to_signature!(kernel, :($(RANGELENGTHS_VARNAMES[3])::$INT_CUDA))
    end
    ranges = [:($RANGES_VARNAME[1]), :($RANGES_VARNAME[2]), :($RANGES_VARNAME[3])]
    if (package == PKG_CUDA)
        body = add_threadids(indices, ranges, body)
        body = literaltypes(numbertype, body)
    elseif (package == PKG_THREADS)
        body = add_loop(indices, ranges, body)
        body = literaltypes(numbertype, body)
    else
        @ArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).")
    end
    check_mask_macro(caller)
    body = apply_masks(body, indices)
    body = add_return(body)
    set_body!(kernel, body)
    return kernel
end


## FUNCTIONS TO DEAL WITH MASKS (@WITHIN) AND INDICES

function check_mask_macro(caller::Module)
    if !isdefined(caller, Symbol("@within")) @MethodPluginError("the macro @within is not defined in the caller. You need to load one of the submodules ParallelStencil.FiniteDifferences{1|2|3}D (or a compatible custom module or set of macros).") end
    methods_str = string(methods(getfield(caller, Symbol("@within"))))
    if !occursin(r"(var\"@within\"|@within)\(__source__::LineNumberNode, __module__::Module, .*::String, .*::Symbol\)", methods_str) @MethodPluginError("the signature of the macro @within is not compatible with ParallelStencil (detected signature: \"$methods_str\"). The signature must correspond to the description in ParallelStencil.WITHIN_DOC. See in ParallelStencil.FiniteDifferences{1|2|3}D for examples.") end
end

function apply_masks(expr::Expr, indices::Array{Any})
    args = expr.args
    for i=1:length(args)
        if typeof(args[i]) == Expr
            e = args[i]
            if e.head == :(=) && typeof(e.args[1]) == Expr && e.args[1].head == :macrocall
                lefthand_macro = e.args[1].args[1]
                lefthand_var   = e.args[1].args[3]
                macroname = string(lefthand_macro)
                args[i] = quote
                              if (@within($macroname, $lefthand_var))
                                  $e
                              end
                          end
            else
                args[i] = apply_masks(e, indices)
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
