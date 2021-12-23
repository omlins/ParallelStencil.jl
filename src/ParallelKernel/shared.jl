# Enable CUDA if the CUDA package is installed (enables to use the package for CPU-only without requiring the CUDA package installed if the installation procedure allows it).
const CUDA_IS_INSTALLED = (Base.find_package("CUDA")!==nothing)
const ENABLE_CUDA = true # NOTE: Can be set to CUDA_IS_INSTALLED, or to true or false independent of it.
const PKG_CUDA = :CUDA
const PKG_THREADS = :Threads
const PKG_NONE = :PKG_NONE
@static if ENABLE_CUDA
    using CUDA
    const SUPPORTED_PACKAGES = [PKG_THREADS, PKG_CUDA]
else
    const SUPPORTED_PACKAGES = [PKG_THREADS]
end
using MacroTools
import MacroTools: postwalk, splitdef, combinedef, isexpr # NOTE: inexpr_walk used instead of MacroTools.inexpr


## CONSTANTS AND TYPES
#NOTE: constants needs to be defined before including the submodules to have them accessible there.

const GENSYM_SEPARATOR = ", "
gensym_world(tag::String, generator::Module) = gensym(string(tag, GENSYM_SEPARATOR, generator)) #NOTE: this function needs to be defind before constants using it.
gensym_world(tag::Symbol, generator::Module) = gensym(string(tag, GENSYM_SEPARATOR, generator))

const INT_CUDA = Int64
const INT_THREADS = Int64
const NTHREADS_MAX = 256
const INDICES = (gensym_world("ix", @__MODULE__), gensym_world("iy", @__MODULE__), gensym_world("iz", @__MODULE__))
const RANGES_VARNAME = gensym_world("ranges", @__MODULE__)
const RANGELENGTHS_VARNAMES = (gensym_world("rangelength_x", @__MODULE__), gensym_world("rangelength_y", @__MODULE__), gensym_world("rangelength_z", @__MODULE__))
const THREADIDS_VARNAMES = (gensym_world("tx", @__MODULE__), gensym_world("ty", @__MODULE__), gensym_world("tz", @__MODULE__))
const RANGES_TYPE_1D = UnitRange{}
const RANGES_TYPE_1D_TUPLE = Tuple{UnitRange{}}
const RANGES_TYPE_2D = Tuple{UnitRange{},UnitRange{}}
const RANGES_TYPE = Tuple{UnitRange{},UnitRange{},UnitRange{}}
const MAXSIZE_TYPE_1D = Integer
const MAXSIZE_TYPE_1D_TUPLE = Tuple{T} where T <: Integer
const MAXSIZE_TYPE_2D = Tuple{T, T} where T <: Integer
const MAXSIZE_TYPE = Tuple{T, T, T} where T <: Integer
const BOUNDARY_WIDTH_TYPE_1D = Integer
const BOUNDARY_WIDTH_TYPE_1D_TUPLE = Tuple{T} where T <: Integer
const BOUNDARY_WIDTH_TYPE_2D = Tuple{T, T} where T <: Integer
const BOUNDARY_WIDTH_TYPE = Tuple{T, T, T} where T <: Integer
const OPERATORS = [:-, :+, :*, :/, :%, :!, :&&, :||] #NOTE: ^ is not contained as causes an error.
const SUPPORTED_LITERALTYPES =      [Float16, Float32, Float64, Complex{Float16}, Complex{Float32}, Complex{Float64}, Int128, Int16, Int32, Int64, Int8, UInt128, UInt16, UInt32, UInt64, UInt8] # NOTE: Not isbitstype as required for CUDA: BigFloat, BigInt, Complex{BigFloat}, Complex{BigInt}
const SUPPORTED_NUMBERTYPES  =      [Float16, Float32, Float64, Complex{Float16}, Complex{Float32}, Complex{Float64}]
const PKNumber               = Union{Float16, Float32, Float64, Complex{Float16}, Complex{Float32}, Complex{Float64}} # NOTE: this always needs to correspond to SUPPORTED_NUMBERTYPES!
const NUMBERTYPE_NONE = DataType
const ERRMSG_UNSUPPORTED_PACKAGE = "unsupported package for parallelization"
const ERRMSG_CHECK_PACKAGE  = "package has to be one of the following: $(join(SUPPORTED_PACKAGES,", "))"
const ERRMSG_CHECK_NUMBERTYPE = "numbertype has to be one of the following: $(join(SUPPORTED_NUMBERTYPES,", "))"
const ERRMSG_CHECK_LITERALTYPES = "the type given to 'literaltype' must be one of the following: $(join(SUPPORTED_LITERALTYPES,", "))"

struct Dim3
    x::INT_THREADS
    y::INT_THREADS
    z::INT_THREADS
end


## FUNCTIONS TO DEAL WITH KERNEL DEFINITIONS: SIGNATURES, BODY AND RETURN STATEMENT

extract_kernel_args(kernel::Expr)   = return (splitdef(kernel)[:args], splitdef(kernel)[:kwargs])
get_body(kernel::Expr)              = return kernel.args[2]
set_body!(kernel::Expr, body::Expr) = ((kernel.args[2] = body); return kernel)

function push_to_signature!(kernel::Expr, arg::Expr)
    kernel_elems = splitdef(kernel)
    push!(kernel_elems[:args], arg)
    kernel = combinedef(kernel_elems)
    return kernel
end

function remove_return(body::Expr)
    if !(body.args[end] in [:(return), :(return nothing), :(nothing)])
        @ArgumentError("invalid kernel in @parallel kernel definition: the last statement must be a `return nothing` statement ('return' or 'return nothing' or 'nothing') as required for any CUDA kernels.")
    end
    remainder = copy(body)
    remainder.args = body.args[1:end-2]
    if inexpr_walk(remainder, :return) @ArgumentError("invalid kernel in @parallel kernel definition: only one return statement is allowed in the kernel and it must return nothing and be the last statement (required to ensure equal behaviour with different packages for parallellization).") end
    return remainder
end

function add_return(body::Expr)
    quote
        $body
        return nothing
    end
end


## FUNCTIONS TO DEAL WITH KERNEL/MACRO CALLS: CHECK IF DEFINITION/CALL, EXTRACT, SPLIT AND EVALUATE ARGUMENTS

is_kernel(arg)      = isdef(arg) # NOTE: to be replaced with MacroTools.isdef(arg): isdef is to be merged fixed in MacroTools (see temporary functions at the end of this file)
is_call(arg)        = ( isa(arg, Expr) && (arg.head == :call) )
is_block(arg)       = ( isa(arg, Expr) && (arg.head == :block) )
is_parallel_call(x) = isexpr(x, :macrocall) && (x.args[1] == Symbol("@parallel") || x.args[1] == :(@parallel))

macro get_args(args...) return args end
extract_args(call::Expr, macroname::Symbol) = eval(substitute(deepcopy(call), macroname, Symbol("@get_args")))

extract_kernelcall_args(call::Expr)         = split_args(call.args[2:end]; in_kernelcall=true)

function is_kwarg(arg; in_kernelcall=false)
    if in_kernelcall return ( isa(arg, Expr) && inexpr_walk(arg, :kw; match_only_head=true) )
    else             return ( isa(arg, Expr) && (arg.head == :(=)) )
    end
end

function split_args(args; in_kernelcall=false)
    posargs   = [x for x in args if !is_kwarg(x; in_kernelcall=in_kernelcall)]
    kwargs    = [x for x in args if  is_kwarg(x; in_kernelcall=in_kernelcall)]
    return posargs, kwargs
end

function split_kwargs(kwargs)
    if !all(is_kwarg.(kwargs)) @ModuleInternalError("not all of kwargs are keyword arguments.") end
    return Dict(x.args[1] => x.args[2] for x in kwargs)
end

function split_parallel_args(args)
    posargs, kwargs = split_args(args[1:end-1])
    kernelarg = args[end]
    if any([x.args[1] in [:blocks, :threads] for x in kwargs]) @KeywordArgumentError("Invalid keyword argument in @parallel call: blocks / threads. They must be passed as positional arguments or been omited.") end
    return posargs, kwargs, kernelarg
end

function eval_arg(caller::Module, arg)
    try
        return @eval(caller, $arg)
    catch e
        @ArgumentEvaluationError("argument $arg could not be evaluated at parse time (in module $caller).")
    end
end


## FUNCTIONS FOR COMMON MANIPULATIONS ON EXPRESSIONS

function substitute(expr::Expr, old, new)
    return postwalk(expr) do x
        if x == old
            return new
        else
            return x;
        end
    end
end

function inexpr_walk(expr::Expr, s::Symbol; match_only_head=false)
    found = false
    postwalk(expr) do x
        if (isa(x,Expr) && x.head==s) found = true end
        if (!match_only_head && (isa(x,Symbol) && x==s)) found = true end
        return x
    end
    return found
end


## FUNCTIONS FOR ERROR HANDLING

check_package(P)                = ( if !isa(P, Symbol) || !(P in SUPPORTED_PACKAGES)  @ArgumentError("$ERRMSG_CHECK_PACKAGE (obtained: $P)." ) end )
check_numbertype(T::DataType)   = ( if !(T in SUPPORTED_NUMBERTYPES) @ArgumentError("$ERRMSG_CHECK_NUMBERTYPE (obtained: $T)." ) end )
check_literaltype(T::DataType)  = ( if !(T in SUPPORTED_LITERALTYPES) @ArgumentError("$ERRMSG_CHECK_LITERALTYPES (obtained: $T)." ) end )
check_numbertype(datatypes...)  = check_numbertype.(datatypes)
check_literaltype(datatypes...) = check_literaltype.(datatypes)


## FUNCTIONS AND MACROS FOR UNIT TESTS

symbols(eval_mod::Union{Symbol,Module}, mod::Union{Symbol,Module}) = @eval(eval_mod, names($mod, all=true, imported=true))
prettystring(expr::Expr)                                           = string(remove_linenumbernodes!(expr))
gorgeousstring(expr::Expr)                                         = string(simplify_varnames!(remove_linenumbernodes!(expr)))
longnameof(f)                                                      = "$(parentmodule(f)).$(nameof(f))"
macro require(condition)               condition_str = string(condition); esc(:( if !($condition) error("pre-test requirement not met: $($condition_str).") end )) end  # Verify a condition required for a unit test (in the unit test results, this should not be treated as a unit test).
macro symbols(eval_mod, mod)           symbols(eval_mod, mod) end
macro macroexpandn(n::Integer, expr)   return QuoteNode(macroexpandn(__module__, expr, n)) end
macro prettyexpand(n::Integer, expr)   return QuoteNode(remove_linenumbernodes!(macroexpandn(__module__, expr, n))) end
macro gorgeousexpand(n::Integer, expr) return QuoteNode(simplify_varnames!(remove_linenumbernodes!(macroexpandn(__module__, expr, n)))) end
macro prettyexpand(expr)               return QuoteNode(remove_linenumbernodes!(macroexpand(__module__, expr; recursive=true))) end
macro gorgeousexpand(expr)             return QuoteNode(simplify_varnames!(remove_linenumbernodes!(macroexpand(__module__, expr; recursive=true)))) end
macro prettystring(args...)            return esc(:(string(ParallelStencil.ParallelKernel.@prettyexpand($(args...))))) end
macro gorgeousstring(args...)          return esc(:(string(ParallelStencil.ParallelKernel.@gorgeousexpand($(args...))))) end

function macroexpandn(m::Module, expr, n::Integer)
    for i = 1:n
        expr = macroexpand(m, expr; recursive=false)
    end
    return expr
end

function remove_linenumbernodes!(expr::Expr)
    expr = Base.remove_linenums!(expr)
    args = expr.args
    for i=1:length(args)
        if isa(args[i], LineNumberNode)
             args[i] = nothing
        elseif typeof(args[i]) == Expr
            args[i] = remove_linenumbernodes!(args[i])
        end
    end
    return expr
end

function simplify_varnames!(expr::Expr)
    args = expr.args
    for i=1:length(args)
        if isa(args[i], Symbol)
            varname = string(args[i]);
            if startswith(varname, "##")
                varname = replace(varname, "##" => "")
                varname = replace(varname, r"#\d*" => "")
                varname = split(varname, GENSYM_SEPARATOR)[1]
                args[i] = Symbol(varname)
            end
        elseif isa(args[i], Expr)
            args[i] = simplify_varnames!(args[i])
        end
    end
    return expr
end


## TEMPORARY FUNCTION DEFINITIONS TO BE MERGED IN MACROTOOLS (https://github.com/FluxML/MacroTools.jl/pull/173)

isdef(ex)     = isshortdef(ex) || islongdef(ex)
islongdef(ex) = @capture(ex, function (fcall_ | fcall_) body_ end)
isshortdef(ex) = MacroTools.isshortdef(ex)
