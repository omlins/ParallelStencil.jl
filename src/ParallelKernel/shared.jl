using CellArrays, StaticArrays, MacroTools
import MacroTools: postwalk, splitdef, combinedef, isexpr, unblock, flatten, rmlines, prewalk # NOTE: inexpr_walk used instead of MacroTools.inexpr

## CONSTANTS AND TYPES (and the macros wrapping them)
# NOTE: constants needs to be defined before including the submodules to have them accessible there.

const GENSYM_SEPARATOR = ", "
gensym_world(tag::String, generator::Module) = gensym(string(tag, GENSYM_SEPARATOR, generator)) #NOTE: this function needs to be defind before constants using it.
gensym_world(tag::Symbol, generator::Module) = gensym(string(tag, GENSYM_SEPARATOR, generator))
gensym_world(tag::Expr,   generator::Module) = gensym(string(tag, GENSYM_SEPARATOR, generator))

ixd(count) = @ModuleInternalError("function ixd had not been evaluated at parse time")
iyd(count) = @ModuleInternalError("function iyd had not been evaluated at parse time")
izd(count) = @ModuleInternalError("function izd had not been evaluated at parse time")

const MOD_METADATA_PK              = gensym_world("__metadata_PK__", @__MODULE__) # # TODO: name mangling should be used here later, or if there is any sense to leave it like that then at check whether it's available must be done before creating it
const PKG_CUDA                     = :CUDA
const PKG_AMDGPU                   = :AMDGPU
const PKG_KERNELABSTRACTIONS       = :KernelAbstractions
const PKG_METAL                    = :Metal
const PKG_THREADS                  = :Threads
const PKG_POLYESTER                = :Polyester
const PKG_NONE                     = :PKG_NONE
const SUPPORTED_PACKAGES           = [PKG_THREADS, PKG_POLYESTER, PKG_CUDA, PKG_AMDGPU, PKG_KERNELABSTRACTIONS, PKG_METAL]
const INT_CUDA                     = Int64 # NOTE: unsigned integers are not yet supported (proper negative offset and range is dealing missing)
const INT_AMDGPU                   = Int64 # NOTE: ...
const INT_KERNELABSTRACTIONS       = Int64 # NOTE: KernelAbstractions dispatch defaults to CPU integers until a GPU-specific handle is selected at runtime.
const INT_METAL                    = Int64 # NOTE: ...
const INT_POLYESTER                = Int64 # NOTE: ...
const INT_THREADS                  = Int64 # NOTE: ...
const COMPUTE_CAPABILITY_DEFAULT   = v"∞" # having it infinity if it is not set allows to directly use statements like `if compute_capability < v"8"`, assuming a recent architecture if it is not set.
const NTHREADS_X_MAX               = 32
const NTHREADS_X_MAX_AMDGPU        = 64
const NTHREADS_X_MAX_KERNELABSTRACTIONS = 32
const NTHREADS_MAX                 = 256
const INDICES                      = (gensym_world("ix", @__MODULE__), gensym_world("iy", @__MODULE__), gensym_world("iz", @__MODULE__))
const INDICES_INN                  = (gensym_world("ixi", @__MODULE__), gensym_world("iyi", @__MODULE__), gensym_world("izi", @__MODULE__)) # ( :($(INDICES[1])+1), :($(INDICES[2])+1), :($(INDICES[3])+1) )
const INDICES_DIR                  = (gensym_world("ixd", @__MODULE__), gensym_world("iyd", @__MODULE__), gensym_world("izd", @__MODULE__))
const INDICES_DIR_FUNCTIONS_SYMS   = (:(ParallelStencil.ParallelKernel.ixd), :(ParallelStencil.ParallelKernel.iyd), :(ParallelStencil.ParallelKernel.izd))
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
const INBOUNDS_DEFAULT             = false
const PADDING_DEFAULT              = false
const MODULENAME_DATA              = :Data
const MODULENAME_TDATA             = :TData
const MODULENAME_DEVICE            = :Device
const MODULENAME_FIELDS            = :Fields
const SCALARTYPES                  = (:Index, :Number, :IndexTuple, :NumberTuple, :IndexCollection, :NumberCollection, :NamedIndexTuple, :NamedNumberTuple)
const ARRAYTYPES                   = (:Array, :Cell, :CellArray, :ArrayTuple, :CellTuple, :CellArrayTuple, :NamedArrayTuple, :NamedCellTuple, :NamedCellArrayTuple, :ArrayCollection, :CellCollection, :CellArrayCollection)
const FIELDTYPES                   = (:Field, :XField, :YField, :ZField, :BXField, :BYField, :BZField, :XXField, :YYField, :ZZField, :XYField, :XZField, :YZField, :VectorField, :BVectorField, :TensorField)
const VECTORNAMES                  = (:x, :y, :z)
const TENSORNAMES                  = (:xx, :yy, :zz, :xy, :xz, :yz)
const AD_MODE_DEFAULT              = :(Enzyme.Reverse)
const AD_DUPLICATE_DEFAULT         = :(Enzyme.DuplicatedNoNeed)
const AD_ANNOTATION_DEFAULT        = :(Enzyme.Const)
const AD_SUPPORTED_ANNOTATIONS     = (Const=:(Enzyme.Const), Active=:(Enzyme.Active), Duplicated=:(Enzyme.Duplicated), DuplicatedNoNeed=:(Enzyme.DuplicatedNoNeed))
const ERRMSG_UNSUPPORTED_PACKAGE   = "unsupported package for parallelization"
const ERRMSG_CHECK_PACKAGE         = "package has to be functional and one of the following: $(join(SUPPORTED_PACKAGES,", "))"
const ERRMSG_CHECK_NUMBERTYPE      = "numbertype has to be one of the following (and evaluatable at parse time): $(join(SUPPORTED_NUMBERTYPES,", "))"
const ERRMSG_CHECK_INBOUNDS        = "inbounds must be a evaluatable at parse time (e.g. literal or constant) and has to be of type Bool."
const ERRMSG_CHECK_PADDING         = "padding must be a evaluatable at parse time (e.g. literal or constant) and has to be of type Bool."
const ERRMSG_CHECK_LITERALTYPES    = "the type given to 'literaltype' must be one of the following: $(join(SUPPORTED_LITERALTYPES,", "))"

const CELLARRAY_BLOCKLENGTH = Dict(PKG_NONE              => 0,
                                   PKG_CUDA              => 0,
                                   PKG_AMDGPU            => 0,
                                   PKG_KERNELABSTRACTIONS => 0,
                                   PKG_METAL             => 0,
                                   PKG_THREADS           => 1,
                                   PKG_POLYESTER         => 1)

struct Dim3
    x::INT_THREADS
    y::INT_THREADS
    z::INT_THREADS
end

macro ranges()       esc(RANGES_VARNAME) end
macro rangelengths() esc(:(($(RANGELENGTHS_VARNAMES...),))) end

function kernel_int_type(package::Symbol)
    if     (package == PKG_CUDA)      int_type = INT_CUDA
    elseif (package == PKG_AMDGPU)    int_type = INT_AMDGPU
    elseif (package == PKG_KERNELABSTRACTIONS) int_type = INT_KERNELABSTRACTIONS
    elseif (package == PKG_METAL)     int_type = INT_METAL
    elseif (package == PKG_THREADS)   int_type = INT_THREADS
    elseif (package == PKG_POLYESTER) int_type = INT_POLYESTER
    end
    return int_type
end

function default_hardware_for(package::Symbol)
    if package == PKG_KERNELABSTRACTIONS
        return :cpu
    elseif package == PKG_CUDA
        return :gpu_cuda
    elseif package == PKG_AMDGPU
        return :gpu_amd
    elseif package == PKG_METAL
        return :gpu_metal
    elseif package == PKG_THREADS || package == PKG_POLYESTER
        return :cpu
    else
        @ArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package). Supported packages are: $(join(SUPPORTED_PACKAGES, ", ")).")
    end
end


## FUNCTIONS TO CHECK EXTENSIONS SUPPORT

is_loaded(arg) = false 
is_installed(package::String) = (Base.find_package(package)!==nothing)


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

function substitute_in_kernel(kernel::Expr, old, new; signature_only::Bool=false, typeparams_only::Bool=false)
    if signature_only
        kernel_elems = splitdef(kernel)
        body = kernel_elems[:body]        # save to restore later
        kernel_elems[:body] = :(return)
        kernel = combinedef(kernel_elems)
    end
    if typeparams_only
        kernel = postwalk(kernel) do ex
            if @capture(ex, type_{typeparams__})
                replace!(typeparams, old => new)
                return :($type{$(typeparams...)})
            else
                return ex
            end
        end
    else
        kernel = substitute(kernel, old, new)
    end
    if signature_only
        kernel_elems = splitdef(kernel)
        kernel_elems[:body] = body
        kernel = combinedef(kernel_elems)
    end
    return kernel
end

function in_signature(kernel::Expr, x)
    kernel_elems = splitdef(kernel)
    kernel_elems[:body] = :()
    signature = combinedef(kernel_elems)
    return inexpr_walk(signature, x)
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

function insert_device_types(caller::Module, kernel::Expr)
    for T in ARRAYTYPES
        if !isnothing(eval_try(caller, :(Data.Device)))
            kernel = substitute(kernel, :(Data.$T), :(Data.Device.$T))
        end
        if !isnothing(eval_try(caller, :(TData.Device)))
            kernel = substitute(kernel, :(TData.$T), :(TData.Device.$T))
        end
    end
    for T in FIELDTYPES
        if !isnothing(eval_try(caller, :(Data.Fields.Device)))
            kernel = substitute(kernel, :(Data.Fields.$T), :(Data.Fields.Device.$T))
        end
        if !isnothing(eval_try(caller, :(TData.Fields.Device)))
            kernel = substitute(kernel, :(TData.Fields.$T), :(TData.Fields.Device.$T))
        end
        Device_val = eval_try(caller, :(Fields.Device))
        if !isnothing(Device_val) && Device_val in (eval_try(caller, :(Data.Fields.Device)), eval_try(caller, :(TData.Fields.Device)))
            kernel = substitute(kernel, :(Fields.$T), :(Fields.Device.$T))
        end
    end
    for T in FIELDTYPES
        T_val = eval_try(caller, T)
        T_d = nothing
        if !isnothing(eval_try(caller, :(Data.Fields.Device)))
            T_d = (!isnothing(T_val) && T_val == eval_try(caller, :(Data.Fields.$T))) ? :(Data.Fields.Device.$T) : T_d
        end
        if !isnothing(eval_try(caller, :(TData.Fields.Device)))
            T_d = (!isnothing(T_val) && T_val == eval_try(caller, :(TData.Fields.$T))) ? :(TData.Fields.Device.$T) : T_d
        end
        if !isnothing(T_d) kernel = substitute_in_kernel(kernel, T, T_d, signature_only=true) end
    end
    return kernel
end

function find_vars(body::Expr, indices::NTuple{N,<:Union{Symbol,Expr}} where N; readonly=false)
    vars         = Dict()
    writevars    = Dict()
    postwalk(body) do ex
        if is_access(ex, indices...)
            @capture(ex, A_[indices_expr__]) || @ModuleInternalError("a indices array access could not be pattern matched.")
            if haskey(vars, A) vars[A] += 1
            else               vars[A]  = 1
            end
        end
        if @capture(ex, (A_[indices_expr__] = rhs_) | (A_[indices_expr__] .= rhs_)) && is_access(:($A[$(indices_expr...)]), indices...)
            if haskey(writevars, A) writevars[A] += 1
            else                    writevars[A]  = 1
            end
        end
        return ex
    end
    if (readonly) return Dict(A => count for (A, count) in vars if A ∉ keys(writevars))
    else          return vars
    end
end

is_access(ex::Expr, ix::Symbol, iy::Symbol, iz::Symbol) = @capture(ex, A_[x_, y_, z_]) && inexpr_walk(x, ix) && inexpr_walk(y, iy) && inexpr_walk(z, iz)
is_access(ex::Expr, ix::Symbol, iy::Symbol)             = @capture(ex, A_[x_, y_])     && inexpr_walk(x, ix) && inexpr_walk(y, iy)
is_access(ex::Expr, ix::Symbol)                         = @capture(ex, A_[x_])         && inexpr_walk(x, ix)
is_access(ex, indices...)                               = false

function is_access(ex::Expr, indices::NTuple{N,<:Union{Symbol,Expr}}, indices_dir::NTuple{N,<:Union{Symbol,Expr}}) where N
    return @capture(ex, A_[ind__]) && length(ind) == N && all(inexpr_walk.(ind, indices) .⊻ inexpr_walk.(ind, indices_dir))
end


## FUNCTIONS TO DEAL WITH KERNEL/MACRO CALLS: CHECK IF DEFINITION/CALL, EXTRACT, SPLIT AND EVALUATE ARGUMENTS

is_kernel(arg)      = isdef(arg) # NOTE: to be replaced with MacroTools.isdef(arg): isdef is to be merged fixed in MacroTools (see temporary functions at the end of this file)
is_call(arg)        = ( isa(arg, Expr) && (arg.head == :call) )
is_block(arg)       = ( isa(arg, Expr) && (arg.head == :block) )
is_parallel_call(x) = isexpr(x, :macrocall) && (x.args[1] == Symbol("@parallel") || x.args[1] == :(@parallel))
is_same(x, y)       = rmlines(x) == rmlines(y) # NOTE: this serves to compare to macros

function extract_args(call::Expr, macroname::Symbol)
    if (call.head != :macrocall) @ModuleInternalError("argument is not a macro call.") end
    if (call.args[1] != macroname) @ModuleInternalError("unexpected macro name.") end
    return (call.args[3:end]...,)
end

extract_kernelcall_args(call::Expr)         = split_args(call.args[2:end]; in_kernelcall=true)
extract_kernelcall_name(call::Expr)         = call.args[1]

function is_kwarg(arg; in_kernelcall=false, separator=:(=), keyword_type=Symbol)
    if in_kernelcall return ( isa(arg, Expr) && inexpr_walk(arg, :kw; match_only_head=true) )
    else             return ( isa(arg, Expr) && (arg.head == separator) && isa(arg.args[1], keyword_type) ) ||
                            ( isa(arg, Expr) && (arg.head == :call) && (arg.args[1] == separator) && isa(arg.args[2], keyword_type) )
    end
end

Base.haskey(::Array{Union{}}, ::Symbol) = return false
    
function Base.haskey(kwargs_expr::Array{Expr}, key::Symbol)
    kwargs = split_kwargs(kwargs_expr)
    return key in keys(kwargs)
end

clean_args(args) = rmlines.(args)

function split_args(args; in_kernelcall=false)
    posargs   = [x for x in args if !is_kwarg(x; in_kernelcall=in_kernelcall)]
    kwargs    = [x for x in args if  is_kwarg(x; in_kernelcall=in_kernelcall)]
    return posargs, kwargs
end

function split_kwargs(kwargs; separator=:(=), keyword_type=Symbol)
    if !all(is_kwarg.(kwargs; separator=separator, keyword_type=keyword_type)) @ModuleInternalError("not all of kwargs are keyword arguments.") end
    return Dict{keyword_type,Any}((x.head==:call) ? (x.args[2] => x.args[3]) : (x.args[1] => x.args[2]) for x in kwargs)
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

function extract_kwargs(caller::Module, kwargs_expr, valid_kwargs, macroname; eval_args=(), separator=:(=), keyword_type=Symbol)
    kwargs_known, = extract_kwargs(caller, kwargs_expr, valid_kwargs, macroname, false; eval_args=eval_args, separator=separator, keyword_type=keyword_type)
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

function eval_try(caller::Module, expr; when_interactive::Bool=true)
    if !when_interactive && isinteractive() # NOTE: this is required to avoid that this function returns non-constant values in interactive sessions, when not appropriate (e.g. in for optimization)
        return nothing
    else
        try
            return @eval(caller, $expr)
        catch e
            return nothing
        end
    end
end


## FUNCTIONS FOR COMMON MANIPULATIONS ON EXPRESSIONS

function substitute(expr::Expr, old, new; inQuoteNode=false, inString=false)
    old_str = string(old)
    new_str = string(new)
    return postwalk(expr) do x
        if x == old
            return new
        elseif inQuoteNode && isa(x, QuoteNode) && x.value == old
            return QuoteNode(new)
        elseif inString && isa(x, String) && occursin(old_str, x)
            return replace(x, old_str => new_str)
        else
            return x;
        end
    end
end

function substitute(expr::Union{Symbol,Expr}, rules::NamedTuple; inQuoteNode=false)
    return postwalk(expr) do x
        if isa(x, Symbol) && haskey(rules, x)
            return rules[x]
        elseif inQuoteNode && isa(x, QuoteNode) && isa(x.value, Symbol) && haskey(rules, x.value)
            return QuoteNode(rules[x.value])
        else
            return x
        end
    end
end

substitute(expr, old, new; inQuoteNode=false, inString=false) = (old == expr) ? new : expr

function increment_arg(expr::Union{Symbol,Expr}, f::Union{Symbol,Expr}; increment::Integer=1)
    return postwalk(expr) do x
        if @capture(x, $f(arg_)) && isa(arg, Integer)
            return :($f($(arg + increment)))
        else
            return x
        end
        # if isa(x, Expr) && (x.head == :call) && length(x.args==2) && (x.args[1] == f) && isa(x.args[2], Integer)
        #     return :($f($(x.args[2] + increment)))
        # else
        #     return x
        # end
    end
end

function promote_to_parent(expr::Union{Symbol,Expr})
    if !@capture(expr, ex_.parent) return :($(expr).parent)
    else                           return expr
    end
end

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
Base.unquoted(b::Bool)   = b

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
check_padding(padding)          = ( if !isa(padding, Bool) @ArgumentError("$ERRMSG_CHECK_INBOUNDS (obtained: $padding)." ) end )


## FUNCTIONS AND MACROS FOR UNIT TESTS

prettystring(expr::Expr)   = string(remove_linenumbernodes!(expr))
gorgeousstring(expr::Expr) = string(simplify_varnames!(remove_linenumbernodes!(expr)))
longnameof(f)              = "$(parentmodule(f)).$(nameof(f))"
symbols(eval_mod::Union{Symbol,Module}, mod::Union{Symbol,Expr,Module}; imported=false, all=true) = @eval(eval_mod, names($mod, all=$all, imported=$imported))
macro symbols(eval_mod, mod, imported=false, all=true) symbols(eval_mod, mod; all=all, imported=imported) end
macro require(condition)               condition_str = string(condition); esc(:( if !($condition) error("pre-test requirement not met: $($condition_str).") end )) end  # Verify a condition required for a unit test (in the unit test results, this should not be treated as a unit test).
macro isgpu(package)                   isgpu(package) end
macro iscpu(package)                   iscpu(package) end
macro macroexpandn(n::Integer, expr)   return QuoteNode(macroexpandn(__module__, expr, n)) end
macro prettyexpand(n::Integer, expr)   return QuoteNode(remove_linenumbernodes!(macroexpandn(__module__, expr, n))) end
macro gorgeousexpand(n::Integer, expr) return QuoteNode(simplify_varnames!(remove_linenumbernodes!(macroexpandn(__module__, expr, n)))) end
macro prettyexpand(expr)               return QuoteNode(remove_linenumbernodes!(macroexpand(__module__, expr; recursive=true))) end
macro gorgeousexpand(expr)             return QuoteNode(simplify_varnames!(remove_linenumbernodes!(macroexpand(__module__, expr; recursive=true)))) end
macro prettystring(args...)            return esc(:(string(ParallelStencil.ParallelKernel.@prettyexpand($(args...))))) end
macro gorgeousstring(args...)          return esc(:(string(ParallelStencil.ParallelKernel.@gorgeousexpand($(args...))))) end
macro interpolate(args...)             esc(interpolate(args...)) end

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

remove_linenumbernodes!(x::Nothing) = x

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


function interpolate(sym::Symbol, vals::NTuple, block::Expr)
    return flatten(unblock(quote
        $((substitute(block, sym, val; inQuoteNode=true, inString=true) for val in vals)...)
    end))
end

interpolate(sym::Symbol, vals_expr::Expr, block::Expr) = interpolate(sym, (extract_tuple(vals_expr)...,), block)

quote_expr(expr) = :($(Expr(:quote, expr)))


## FUNCTIONS TO QUERY DEVICE PROPERTIES

function get_compute_capability(package::Symbol)
    default = COMPUTE_CAPABILITY_DEFAULT
    if     (package == PKG_CUDA)              get_cuda_compute_capability(default)
    elseif (package == PKG_AMDGPU)            get_amdgpu_compute_capability(default)
    elseif (package == PKG_KERNELABSTRACTIONS) get_cpu_compute_capability(default)
    elseif (package == PKG_METAL)             get_metal_compute_capability(default)
    elseif (package == PKG_THREADS)           get_cpu_compute_capability(default)
    elseif (package == PKG_POLYESTER)         get_cpu_compute_capability(default)
    else
        @ArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package). Supported packages are: $(join(SUPPORTED_PACKAGES, ", ")).")
    end
end

get_cpu_compute_capability(default::VersionNumber) = return default


## FUNCTIONS/MACROS FOR DIVERSE SYNTAX SUGAR

iscpu(package) = return (package in (PKG_THREADS, PKG_POLYESTER))
isgpu(package) = return (package in (PKG_CUDA, PKG_AMDGPU, PKG_METAL))

hasmeta_PK(caller::Module) = isdefined(caller, MOD_METADATA_PK)


## TEMPORARY FUNCTION DEFINITIONS TO BE MERGED IN MACROTOOLS (https://github.com/FluxML/MacroTools.jl/pull/173)

isdef(ex)     = isshortdef(ex) || islongdef(ex)
islongdef(ex) = @capture(ex, function (fcall_ | fcall_) body_ end)
isshortdef(ex) = MacroTools.isshortdef(ex)
