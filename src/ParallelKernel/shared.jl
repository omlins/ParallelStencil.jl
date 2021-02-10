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
import MacroTools: postwalk, isexpr, inexpr


## CONSTANTS

const GENSYM_SEPARATOR = ", "
gensym_world(tag::String, generator::Module) = gensym(string(tag, GENSYM_SEPARATOR, generator)) #NOTE: this function needs to be defind before constants using it.
gensym_world(tag::Symbol, generator::Module) = gensym(string(tag, GENSYM_SEPARATOR, generator))

const INT_CUDA = Int64
const NTHREADS_MAX = 256
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
const SUPPORTED_LITERALTYPES = [Float16, Float32, Float64, Complex{Float16}, Complex{Float32}, Complex{Float64}, Int128, Int16, Int32, Int64, Int8, UInt128, UInt16, UInt32, UInt64, UInt8] # NOTE: Not isbitstype as required for CUDA: BigFloat, BigInt, Complex{BigFloat}, Complex{BigInt}
const SUPPORTED_NUMBERTYPES  = [Float16, Float32, Float64, Complex{Float16}, Complex{Float32}, Complex{Float64}]
const NUMBERTYPE_NONE = DataType
const ERRMSG_UNSUPPORTED_PACKAGE = "unsupported package for parallelization"
const ERRMSG_CHECK_PACKAGE  = "package has to be one of the following: $(join(SUPPORTED_PACKAGES,", "))"
const ERRMSG_CHECK_NUMBERTYPE = "numbertype has to be one of the following: $(join(SUPPORTED_NUMBERTYPES,", "))"
const ERRMSG_CHECK_LITERALTYPES = "the type given to 'literaltype' must be one of the following: $(join(SUPPORTED_LITERALTYPES,", "))"


## FUNCTIONS TO DEAL WITH FUNCTION DEFINITIONS: SIGNATURES, BODY AND RETURN STATEMENT

function push_to_signature!(kernel::Expr, arg::Expr)
    push!(kernel.args[1].args, arg)
    return kernel
end

function get_body(kernel::Expr)
    return kernel.args[2]
end

function set_body!(kernel::Expr, body::Expr)
    kernel.args[2] = body
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


## FUNCTIONS TO DEAL WITH FUNCTION/MACRO CALLS: CHECK IF DEFINITION/CALL, EXTRACT, SPLIT AND EVALUATE ARGUMENTS

is_function(arg)    = ( isa(arg, Expr) && ( (arg.head == :function) || (arg.head == :(=) && isa(arg.args[1], Expr) && arg.args[1].head == :call) ) )
is_call(arg)        = ( isa(arg, Expr) && (arg.head == :call) )
is_block(arg)       = ( isa(arg, Expr) && (arg.head == :block) )
is_parallel_call(x) = isexpr(x, :macrocall) && (x.args[1] == Symbol("@parallel") || x.args[1] == :(@parallel))

macro get_args(args...) return args end
extract_args(call::Expr, macroname::Symbol) = eval(substitute(deepcopy(call), macroname, Symbol("@get_args")))

function split_args(args)
    posargs   = [x for x in args[1:end-1] if !(isa(x,Expr) && x.head == :(=))]
    kwargs    = [x for x in args[1:end-1] if isa(x,Expr) && x.head == :(=)]
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
    postwalk(expr) do x
        if x == old
            return new
        else
            return x;
        end
    end
end

function inexpr_walk(expr::Expr, s::Symbol)
    found = false
    postwalk(expr) do x
        if ((isa(x,Expr) && x.head==s) || (isa(x,Symbol) && x==s)) found = true end
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
