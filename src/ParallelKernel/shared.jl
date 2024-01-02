# Enable CUDA/AMDGPU if the CUDA/AMDGPU package is installed or in any case (enables to use the package for CPU-only without requiring the CUDA/AMDGPU package installed if the installation procedure allows it).
const CUDA_IS_INSTALLED   = (Base.find_package("CUDA")!==nothing)
const AMDGPU_IS_INSTALLED = (Base.find_package("AMDGPU")!==nothing)
const ENABLE_CUDA         = true # NOTE: Can be set to CUDA_IS_INSTALLED, or to true or false independent of it.
const ENABLE_AMDGPU       = true # NOTE: Can be set to AMDGPU_IS_INSTALLED, or to true or false independent of it.
const PKG_CUDA            = :CUDA
const PKG_AMDGPU          = :AMDGPU
const PKG_THREADS         = :Threads
const PKG_NONE            = :PKG_NONE
@static if ENABLE_CUDA && ENABLE_AMDGPU
    using CUDA
    using AMDGPU
    const SUPPORTED_PACKAGES = [PKG_THREADS, PKG_CUDA, PKG_AMDGPU]
elseif ENABLE_CUDA 
    using CUDA
    const SUPPORTED_PACKAGES = [PKG_THREADS, PKG_CUDA]
elseif ENABLE_AMDGPU
    using AMDGPU
    const SUPPORTED_PACKAGES = [PKG_THREADS, PKG_AMDGPU]
else
    const SUPPORTED_PACKAGES = [PKG_THREADS]
end
import Enzyme
using CellArrays, StaticArrays, MacroTools
import MacroTools: postwalk, splitdef, combinedef, isexpr, unblock # NOTE: inexpr_walk used instead of MacroTools.inexpr


## CONSTANTS AND TYPES (and the macros wrapping them)
#NOTE: constants needs to be defined before including the submodules to have them accessible there.

const GENSYM_SEPARATOR = ", "
gensym_world(tag::String, generator::Module) = gensym(string(tag, GENSYM_SEPARATOR, generator)) #NOTE: this function needs to be defind before constants using it.
gensym_world(tag::Symbol, generator::Module) = gensym(string(tag, GENSYM_SEPARATOR, generator))
gensym_world(tag::Expr,   generator::Module) = gensym(string(tag, GENSYM_SEPARATOR, generator))

const INT_CUDA                     = Int64 # NOTE: unsigned integers are not yet supported (proper negative offset and range is dealing missing)
const INT_AMDGPU                   = Int64 # NOTE: ...
const INT_THREADS                  = Int64 # NOTE: ...
const NTHREADS_MAX                 = 256
const INDICES                      = (gensym_world("ix", @__MODULE__), gensym_world("iy", @__MODULE__), gensym_world("iz", @__MODULE__))
const RANGES_VARNAME               = gensym_world("ranges", @__MODULE__)
const RANGELENGTHS_VARNAMES        = (gensym_world("rangelength_x", @__MODULE__), gensym_world("rangelength_y", @__MODULE__), gensym_world("rangelength_z", @__MODULE__))
const THREADIDS_VARNAMES           = (gensym_world("tx", @__MODULE__), gensym_world("ty", @__MODULE__), gensym_world("tz", @__MODULE__))
const RANGES_TYPE_1D               = UnitRange{}
const RANGES_TYPE_1D_TUPLE         = Tuple{UnitRange{}}
const RANGES_TYPE_2D               = Tuple{UnitRange{},UnitRange{}}
const RANGES_TYPE                  = Tuple{UnitRange{},UnitRange{},UnitRange{}}
const RANGELENGTH_XYZ_TYPE         = T where T <: Integer
const MAXSIZE_TYPE_1D              = Integer
const MAXSIZE_TYPE_1D_TUPLE        = Tuple{T} where T <: Integer
const MAXSIZE_TYPE_2D              = Tuple{T, T} where T <: Integer
const MAXSIZE_TYPE                 = Tuple{T, T, T} where T <: Integer
const BOUNDARY_WIDTH_TYPE_1D       = Integer
const BOUNDARY_WIDTH_TYPE_1D_TUPLE = Tuple{T} where T <: Integer
const BOUNDARY_WIDTH_TYPE_2D       = Tuple{T, T} where T <: Integer
const BOUNDARY_WIDTH_TYPE          = Tuple{T, T, T} where T <: Integer
const INDEXING_OPERATORS           = [:-, :+, :*, :÷, :%, :<, :>, :<=, :>=, :(==), :(!=), :(:), :!, :&&, :||] #NOTE: ^ is not contained as causes an error. :(:),
const SUPPORTED_LITERALTYPES       =      [Float16, Float32, Float64, Complex{Float16}, Complex{Float32}, Complex{Float64}, Int128, Int16, Int32, Int64, Int8, UInt128, UInt16, UInt32, UInt64, UInt8] # NOTE: Not isbitstype as required for CUDA: BigFloat, BigInt, Complex{BigFloat}, Complex{BigInt}
const SUPPORTED_NUMBERTYPES        =      [Float16, Float32, Float64, Complex{Float16}, Complex{Float32}, Complex{Float64}]
const PKNumber                     = Union{Float16, Float32, Float64, Complex{Float16}, Complex{Float32}, Complex{Float64}} # NOTE: this always needs to correspond to SUPPORTED_NUMBERTYPES!
const NUMBERTYPE_NONE              = DataType
const AD_MODE_DEFAULT              = :(Enzyme.Reverse)
const AD_DUPLICATE_DEFAULT         = Enzyme.DuplicatedNoNeed
const AD_ANNOTATION_DEFAULT        = Enzyme.Const
const AD_SUPPORTED_ANNOTATIONS     = (Const=Enzyme.Const, Active=Enzyme.Active, Duplicated=Enzyme.Duplicated, DuplicatedNoNeed=Enzyme.DuplicatedNoNeed)
const ERRMSG_UNSUPPORTED_PACKAGE   = "unsupported package for parallelization"
const ERRMSG_CHECK_PACKAGE         = "package has to be functional and one of the following: $(join(SUPPORTED_PACKAGES,", "))"
const ERRMSG_CHECK_NUMBERTYPE      = "numbertype has to be one of the following (and evaluatable at parse time): $(join(SUPPORTED_NUMBERTYPES,", "))"
const ERRMSG_CHECK_INBOUNDS        = "inbounds must be a evaluatable at parse time (e.g. literal or constant) and has to be of type Bool."
const ERRMSG_CHECK_LITERALTYPES    = "the type given to 'literaltype' must be one of the following: $(join(SUPPORTED_LITERALTYPES,", "))"

const CELLARRAY_BLOCKLENGTH = Dict(PKG_NONE    => 0,
                                   PKG_CUDA    => 0,
                                   PKG_AMDGPU  => 0,
                                   PKG_THREADS => 1)

struct Dim3
    x::INT_THREADS
    y::INT_THREADS
    z::INT_THREADS
end

macro ranges()       esc(RANGES_VARNAME) end
macro rangelengths() esc(:(($(RANGELENGTHS_VARNAMES...),))) end


## FUNCTIONS TO GET CREATE AND MANAGE CUDA STREAMS, AMDGPU QUEUES AND "ROCSTREAMS"

@static if ENABLE_CUDA
    let
        global get_priority_custream, get_custream
        priority_custreams = Array{CuStream}(undef, 0)
        custreams          = Array{CuStream}(undef, 0)

        function get_priority_custream(id::Integer)
            while (id > length(priority_custreams)) push!(priority_custreams, CuStream(; flags=CUDA.STREAM_NON_BLOCKING, priority=CUDA.priority_range()[end])) end # CUDA.priority_range()[end] is max priority. # NOTE: priority_range cannot be called outside the function as only at runtime sure that CUDA is functional.
            return priority_custreams[id]
        end

        function get_custream(id::Integer)
            while (id > length(custreams)) push!(custreams, CuStream(; flags=CUDA.STREAM_NON_BLOCKING, priority=CUDA.priority_range()[1])) end # CUDA.priority_range()[1] is min priority. # NOTE: priority_range cannot be called outside the function as only at runtime sure that CUDA is functional.
            return custreams[id]
        end
    end
end

@static if ENABLE_AMDGPU
    let
        global get_priority_rocstream, get_rocstream
        priority_rocstreams = Array{AMDGPU.HIPStream}(undef, 0)
        rocstreams          = Array{AMDGPU.HIPStream}(undef, 0)

        function get_priority_rocstream(id::Integer)
            while (id > length(priority_rocstreams)) push!(priority_rocstreams, AMDGPU.HIPStream(:high)) end
            return priority_rocstreams[id]
        end

        function get_rocstream(id::Integer)
            while (id > length(rocstreams)) push!(rocstreams, AMDGPU.HIPStream(:low)) end
            return rocstreams[id]
        end
    end
end


## FUNCTIONS TO DEAL WITH KERNEL DEFINITIONS: SIGNATURES, BODY AND RETURN STATEMENT

extract_kernel_args(kernel::Expr)     = return (splitdef(kernel)[:args], splitdef(kernel)[:kwargs])
get_body(kernel::Expr)                = return kernel.args[2]
set_body!(kernel::Expr, body::Expr)   = ((kernel.args[2] = body); return kernel)
get_name(kernel::Expr)                = return splitdef(kernel)[:name]

function set_name(kernel::Expr, name::Symbol)
    kernel_elems = splitdef(kernel)
    kernel_elems[:name] = name
    kernel = combinedef(kernel_elems)
    return kernel
end

function push_to_signature!(kernel::Expr, arg::Expr)
    kernel_elems = splitdef(kernel)
    push!(kernel_elems[:args], arg)
    kernel = combinedef(kernel_elems)
    return kernel
end

function remove_return(body::Expr)
    if !(body.args[end] in [:(return), :(return nothing), :(nothing)])
        @ArgumentError("invalid kernel in @parallel kernel definition: the last statement must be a `return nothing` statement ('return' or 'return nothing' or 'nothing') as required for any GPU kernels.")
    end
    body = disguise_nested_returns(body)
    remainder = copy(body)
    remainder.args = body.args[1:end-2]
    if inexpr_walk(remainder, :return) @ArgumentError("invalid kernel in @parallel kernel definition: only one return statement is allowed in the kernel (exception: nested function definitions) and it must return nothing and be the last statement (required to ensure equal behaviour with different packages for parallellization).") end
    return remainder
end

function disguise_nested_returns(body::Expr)
    return postwalk(body) do ex
        if isdef(ex)
            f_elems = splitdef(ex)
            body = f_elems[:body]
            f_elems[:body] = disguise_returns(body)
            return combinedef(f_elems)
        else
            return ex
        end
    end
end

function disguise_returns(body::Expr)
    return postwalk(body) do ex
        if @capture(ex, return x_)
            return :(ParallelStencil.ParallelKernel.@return_value($x))
        elseif @capture(ex, return)
            return :(ParallelStencil.ParallelKernel.@return_nothing)
        else
            return ex
        end
    end
end

function add_return(body::Expr)
    quote
        $body
        return nothing
    end
end

function add_inbounds(body::Expr)
    quote
        Base.@inbounds begin
            $body
        end
    end
end

function insert_device_types(kernel::Expr)
    kernel = substitute(kernel, :(Data.Array),                :(Data.DeviceArray))
    kernel = substitute(kernel, :(Data.Cell),                 :(Data.DeviceCell))
    kernel = substitute(kernel, :(Data.CellArray),            :(Data.DeviceCellArray))
    kernel = substitute(kernel, :(Data.ArrayTuple),           :(Data.DeviceArrayTuple))
    kernel = substitute(kernel, :(Data.CellTuple),            :(Data.DeviceCellTuple))
    kernel = substitute(kernel, :(Data.CellArrayTuple),       :(Data.DeviceCellArrayTuple))
    kernel = substitute(kernel, :(Data.NamedArrayTuple),      :(Data.NamedDeviceArrayTuple))
    kernel = substitute(kernel, :(Data.NamedCellTuple),       :(Data.NamedDeviceCellTuple))
    kernel = substitute(kernel, :(Data.NamedCellArrayTuple),  :(Data.NamedDeviceCellArrayTuple))
    kernel = substitute(kernel, :(Data.ArrayCollection),      :(Data.DeviceArrayCollection))
    kernel = substitute(kernel, :(Data.CellCollection),       :(Data.DeviceCellCollection))
    kernel = substitute(kernel, :(Data.CellArrayCollection),  :(Data.DeviceCellArrayCollection))
    kernel = substitute(kernel, :(Data.TArray),               :(Data.DeviceTArray))
    kernel = substitute(kernel, :(Data.TCell),                :(Data.DeviceTCell))
    kernel = substitute(kernel, :(Data.TCellArray),           :(Data.DeviceTCellArray))
    kernel = substitute(kernel, :(Data.TArrayTuple),          :(Data.DeviceTArrayTuple))
    kernel = substitute(kernel, :(Data.TCellTuple),           :(Data.DeviceTCellTuple))
    kernel = substitute(kernel, :(Data.TCellArrayTuple),      :(Data.DeviceTCellArrayTuple))
    kernel = substitute(kernel, :(Data.NamedTArrayTuple),     :(Data.NamedDeviceTArrayTuple))
    kernel = substitute(kernel, :(Data.NamedTCellTuple),      :(Data.NamedDeviceTCellTuple))
    kernel = substitute(kernel, :(Data.NamedTCellArrayTuple), :(Data.NamedDeviceTCellArrayTuple))
    kernel = substitute(kernel, :(Data.TArrayCollection),     :(Data.DeviceTArrayCollection))
    kernel = substitute(kernel, :(Data.TCellCollection),      :(Data.DeviceTCellCollection))
    kernel = substitute(kernel, :(Data.TCellArrayCollection), :(Data.DeviceTCellArrayCollection))
    return kernel
end



## FUNCTIONS TO DEAL WITH KERNEL/MACRO CALLS: CHECK IF DEFINITION/CALL, EXTRACT, SPLIT AND EVALUATE ARGUMENTS

is_kernel(arg)      = isdef(arg) # NOTE: to be replaced with MacroTools.isdef(arg): isdef is to be merged fixed in MacroTools (see temporary functions at the end of this file)
is_call(arg)        = ( isa(arg, Expr) && (arg.head == :call) )
is_block(arg)       = ( isa(arg, Expr) && (arg.head == :block) )
is_parallel_call(x) = isexpr(x, :macrocall) && (x.args[1] == Symbol("@parallel") || x.args[1] == :(@parallel))

function extract_args(call::Expr, macroname::Symbol)
    if (call.head != :macrocall) @ModuleInternalError("argument is not a macro call.") end
    if (call.args[1] != macroname) @ModuleInternalError("unexpected macro name.") end
    return (call.args[3:end]...,)
end

extract_kernelcall_args(call::Expr)         = split_args(call.args[2:end]; in_kernelcall=true)
extract_kernelcall_name(call::Expr)         = call.args[1]

function is_kwarg(arg; in_kernelcall=false, separator=:(=), keyword_type=Symbol)
    if in_kernelcall return ( isa(arg, Expr) && inexpr_walk(arg, :kw; match_only_head=true) )
    else             return ( isa(arg, Expr) && (arg.head == separator) && isa(arg.args[1], keyword_type))
    end
end

Base.haskey(::Array{Union{}}, ::Symbol) = return false
    
function Base.haskey(kwargs_expr::Array{Expr}, key::Symbol)
    kwargs = split_kwargs(kwargs_expr)
    return key in keys(kwargs)
end

function split_args(args; in_kernelcall=false)
    posargs   = [x for x in args if !is_kwarg(x; in_kernelcall=in_kernelcall)]
    kwargs    = [x for x in args if  is_kwarg(x; in_kernelcall=in_kernelcall)]
    return posargs, kwargs
end

function split_kwargs(kwargs; separator=:(=), keyword_type=Symbol)
    if !all(is_kwarg.(kwargs; separator=separator, keyword_type=keyword_type)) @ModuleInternalError("not all of kwargs are keyword arguments.") end
    return Dict{keyword_type,Any}(x.args[1] => x.args[2] for x in kwargs)
end

function validate_kwargkeys(kwargs::Dict, valid_kwargs::Tuple, macroname::String)
    for k in keys(kwargs)
        if !(k in valid_kwargs) @KeywordArgumentError("Invalid keyword argument in $macroname call: `$k`. Valid keyword arguments are: `$(join(valid_kwargs, "`, `"))`.") end
    end
end

function extract_values(kwargs::Dict, valid_kwargs::Tuple)
    return ((k in keys(kwargs)) ? kwargs[k] : nothing for k in valid_kwargs)
end

function extract_kwargvalues(kwargs_expr, valid_kwargs, macroname)
    kwargs = split_kwargs(kwargs_expr)
    validate_kwargkeys(kwargs, valid_kwargs, macroname)
    return extract_values(kwargs, valid_kwargs)
end

function extract_kwargs(caller::Module, kwargs_expr, valid_kwargs, macroname, has_unknown_kwargs; eval_args=(), separator=:(=), keyword_type=Symbol)
    kwargs = split_kwargs(kwargs_expr, separator=separator, keyword_type=keyword_type)
    if (!has_unknown_kwargs) validate_kwargkeys(kwargs, valid_kwargs, macroname) end
    for k in keys(kwargs)
        if (k in eval_args) kwargs[k] = eval_arg(caller, kwargs[k]) end
    end
    kwargs_known        = NamedTuple(filter(x -> x.first ∈ valid_kwargs, kwargs))
    kwargs_unknown      = (keyword_type == Symbol) ? NamedTuple(filter(x -> x.first ∉ valid_kwargs, kwargs)) : NamedTuple()
    kwargs_unknown_dict = Dict(filter(x -> x.first ∉ valid_kwargs, kwargs))
    kwargs_unknown_expr = [:($k = $(kwargs_unknown_dict[k])) for k in keys(kwargs_unknown_dict)]
    return kwargs_known, kwargs_unknown_expr, kwargs_unknown, kwargs_unknown_dict
end

function extract_kwargs(caller::Module, kwargs_expr, valid_kwargs, macroname; eval_args=())
    kwargs_known, = extract_kwargs(caller, kwargs_expr, valid_kwargs, macroname, false; eval_args=eval_args)
    return kwargs_known
end

function split_parallel_args(args; is_call=true)
    posargs, kwargs = split_args(args[1:end-1])
    kernelarg = args[end]
    if (is_call && any([x.args[1] in [:blocks, :threads] for x in kwargs])) @KeywordArgumentError("Invalid keyword argument in @parallel <kernelcall>: blocks / threads. They must be passed as positional arguments or been omited.") end
    if (is_call && any([x.args[1] in [:groupsize, :gridsize] for x in kwargs])) @KeywordArgumentError("Invalid keyword argument in @parallel <kernelcall>: groupsize / gridsize. CUDA nomenclature and concepts are to be used for @parallel calls (and kernels).") end
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

substitute(expr, old, new) = (old == expr) ? new : expr

function cast(expr::Expr, f::Symbol, type::DataType)
    return postwalk(expr) do ex
        if @capture(ex, $f(args__))
            return :($type($ex))
        else
            return ex
        end
    end
end

function inexpr_walk(expr::Expr, e::Expr)
    found = false
    postwalk(expr) do x
        if (isa(x,Expr) && x==e) found = true end
        return x
    end
    return found
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

inexpr_walk(expr::Symbol, s::Symbol; match_only_head=false) = (s == expr)
inexpr_walk(expr,         s::Symbol; match_only_head=false) = false
inexpr_walk(expr,         e::Expr)                          = false

Base.unquoted(s::Symbol) = s

function extract_tuple(t::Union{Expr,Symbol}; nested=false) # NOTE: this could return a tuple, but would require to change all small arrays to tuples...
    if isa(t, Expr) && t.head == :tuple
        if (nested) return t.args
        else        return Base.unquoted.(t.args)
        end
    else 
        return [t]
    end
end


## FUNCTIONS FOR ERROR HANDLING

check_package(P)                = ( if !isa(P, Symbol) || !(P in SUPPORTED_PACKAGES)  @ArgumentError("$ERRMSG_CHECK_PACKAGE (obtained: $P)." ) end )
check_numbertype(T::DataType)   = ( if !(T in SUPPORTED_NUMBERTYPES) @ArgumentError("$ERRMSG_CHECK_NUMBERTYPE (obtained: $T)." ) end )
check_literaltype(T::DataType)  = ( if !(T in SUPPORTED_LITERALTYPES) @ArgumentError("$ERRMSG_CHECK_LITERALTYPES (obtained: $T)." ) end )
check_numbertype(datatypes...)  = check_numbertype.(datatypes)
check_literaltype(datatypes...) = check_literaltype.(datatypes)
check_inbounds(inbounds)        = ( if !isa(inbounds, Bool) @ArgumentError("$ERRMSG_CHECK_INBOUNDS (obtained: $inbounds)." ) end )


## FUNCTIONS AND MACROS FOR UNIT TESTS

symbols(eval_mod::Union{Symbol,Module}, mod::Union{Symbol,Module}) = @eval(eval_mod, names($mod, all=true, imported=true))
prettystring(expr::Expr)                                           = string(remove_linenumbernodes!(expr))
gorgeousstring(expr::Expr)                                         = string(simplify_varnames!(remove_linenumbernodes!(expr)))
longnameof(f)                                                      = "$(parentmodule(f)).$(nameof(f))"
macro require(condition)               condition_str = string(condition); esc(:( if !($condition) error("pre-test requirement not met: $($condition_str).") end )) end  # Verify a condition required for a unit test (in the unit test results, this should not be treated as a unit test).
macro symbols(eval_mod, mod)           symbols(eval_mod, mod) end
macro isgpu(package)                   isgpu(package) end
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
            if startswith(varname, "@##") || startswith(varname, "##")
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


## FUNCTIONS/MACROS FOR DIVERSE SYNTAX SUGAR

isgpu(package) = return (package in (PKG_CUDA, PKG_AMDGPU))


## TEMPORARY FUNCTION DEFINITIONS TO BE MERGED IN MACROTOOLS (https://github.com/FluxML/MacroTools.jl/pull/173)

isdef(ex)     = isshortdef(ex) || islongdef(ex)
islongdef(ex) = @capture(ex, function (fcall_ | fcall_) body_ end)
isshortdef(ex) = MacroTools.isshortdef(ex)
