#TODO: add ParallelStencil.ParallelKernel. in front of all kernel lang in macros! Later: generalize more for z?

##
macro nx_l(args...) check_initialized(); checksinglearg(args...); esc(nx_l(args...)); end


##
macro ny_l(args...) check_initialized(); checksinglearg(args...); esc(ny_l(args...)); end


##
macro nz_l(args...) check_initialized(); checksinglearg(args...); esc(nz_l(args...)); end


##
macro t_h(args...) check_initialized(); checknoargs(args...); esc(t_h(args...)); end


##
macro t_h2(args...) check_initialized(); checktwoargs(args...); esc(t_h2(args...)); end


##
macro tx_h(args...) check_initialized(); checksinglearg(args...); esc(tx_h(args...)); end


##
macro ty_h(args...) check_initialized(); checksinglearg(args...); esc(ty_h(args...)); end


##
macro tx_h2(args...) check_initialized(); checktwoargs(args...); esc(tx_h2(args...)); end


##
macro ty_h2(args...) check_initialized(); checktwoargs(args...); esc(ty_h2(args...)); end


##
macro ix_h(args...) check_initialized(); checksinglearg(args...); esc(ix_h(args...)); end


##
macro iy_h(args...) check_initialized(); checktwoargs(args...); esc(iy_h(args...)); end


##
macro ix_h2(args...) check_initialized(); checktwoargs(args...); esc(ix_h2(args...)); end


##
macro iy_h2(args...) check_initialized(); checktwoargs(args...); esc(iy_h2(args...)); end


##
macro loop(args...) check_initialized(); checkargs_loop(args...); esc(loop(args...)); end


##
macro loopopt(args...) check_initialized(); checkargs_loopopt(args...); esc(loopopt(__module__, args...)); end


##
macro shortif(args...) check_initialized(); checktwoargs(args...); esc(shortif(args...)); end


## ARGUMENT CHECKS

function checknoargs(args...)
    if (length(args) != 0) @ArgumentError("no arguments allowed.") end
end

function checksinglearg(args...)
    if (length(args) != 1) @ArgumentError("wrong number of arguments.") end
end

function checktwoargs(args...)
    if (length(args) != 2) @ArgumentError("wrong number of arguments.") end
end

function checkargs_loop(args...)
    if (length(args) != 4) @ArgumentError("wrong number of arguments.") end
end

function checkargs_loopopt(args...)
    if (length(args) != 7 && length(args) != 6 && length(args) != 3) @ArgumentError("wrong number of arguments.") end
end


## FUNCTIONS FOR INDEXING AND DIMENSIONS

function nx_l(hx; package::Symbol=get_package())
    if (package ∉ SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    return :(@blockDim().x + (2*$hx))
end

function ny_l(hy; package::Symbol=get_package())
    if (package ∉ SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    return :(@blockDim().y + (2*$hy))
end

function nz_l(hz; package::Symbol=get_package())
    if (package ∉ SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    return :(@blockDim().z + (2*$hz))
end

function t_h(args...; package::Symbol=get_package())
    if (package ∉ SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    return :((@threadIdx().y-1)*@blockDim().x + @threadIdx().x)
end

function t_h2(hx, hy; package::Symbol=get_package())
    if (package ∉ SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    return :(ParallelStencil.@t_h() + ParallelStencil.@nx_l($hx)*ParallelStencil.@ny_l($hy) - @blockDim().x*@blockDim().y)
end

function tx_h(hx; package::Symbol=get_package())
    if (package ∉ SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    return :((ParallelStencil.@t_h() -1) % ParallelStencil.@nx_l($hx) + 1)
end

function ty_h(hx; package::Symbol=get_package())
    if (package ∉ SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    return :((ParallelStencil.@t_h() -1) ÷ ParallelStencil.@nx_l($hx) + 1)
end

function tx_h2(hx, hy; package::Symbol=get_package())
    if (package ∉ SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    return :((ParallelStencil.@t_h2($hx, $hy)-1) % ParallelStencil.@nx_l($hx) + 1)
end

function ty_h2(hx, hy; package::Symbol=get_package())
    if (package ∉ SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    return :((ParallelStencil.@t_h2($hx, $hy)-1) ÷ ParallelStencil.@nx_l($hx) + 1)
end

function ix_h(hx; package::Symbol=get_package())
    if (package ∉ SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    return :((@blockIdx().x-1)*@blockDim().x + ParallelStencil.@tx_h($hx)  - $hx)
end

function iy_h(hx, hy; package::Symbol=get_package())
    if (package ∉ SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    return :((@blockIdx().y-1)*@blockDim().y + ParallelStencil.@ty_h($hx)  - $hy)
end

function ix_h2(hx, hy; package::Symbol=get_package())
    if (package ∉ SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    return :((@blockIdx().x-1)*@blockDim().x + ParallelStencil.@tx_h2($hx, $hy) - $hx)
end

function iy_h2(hx, hy; package::Symbol=get_package())
    if (package ∉ SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    return :((@blockIdx().y-1)*@blockDim().y + ParallelStencil.@ty_h2($hx, $hy) - $hy)
end


## FUNCTIONS FOR PERFORMANCE OPTIMSATIONS

function loop(index::Symbol, optdim::Integer, loopsize, body; package::Symbol=get_package())
    if (package ∉ SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    dimvar = (:x,:y,:z)[optdim]
    loopoffset = gensym_world("loopoffset", @__MODULE__)
    i          = gensym_world("i", @__MODULE__)
    return quote
        $loopoffset = (@blockIdx().$dimvar-1)*$loopsize
        for $i = 1:$loopsize
            $index = $i + $loopoffset
            $body
        end
    end
end


#function add_loopopt(body::Expr, indices::Array{<:Union{Expr,Symbol}}, optvars::Union{Expr,Symbol}, optdim::Integer, loopsize::Union{Expr,Symbol,Integer}, halosize::Union{Integer,NTuple{N,Integer} where N})
#TODO: see what to do with global consts as SHMEM_HALO_X,... Support later multiple vars for opt (now just A=T...)
#TODO: add input check and errors
#TODO: maybe gensym with macro @gensym
function loopopt(caller::Module, indices, optvars, optdim::Integer, loopsize, halosize, indices_shift, body; package::Symbol=get_package())
    if !isa(optvars, Symbol) @KeywordArgumentError("at present, only one optvar is supported.") end
    A = optvars 
    if (package ∉ SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    if isa(indices,Expr) indices = indices.args else indices = (indices,) end
    halosize = eval_arg(caller, halosize)
    noexpr = :(begin end)
    if optdim == 3
        ranges         = RANGES_VARNAME
        rangelength_z  = RANGELENGTHS_VARNAMES[3]
        tz_g           = THREADIDS_VARNAMES[3]
        hx, hy         = halosize
        shmem          = (hx>0 || hy>0)
        _ix, _iy, _iz  = indices
        _i             = gensym_world("i", @__MODULE__)
        ix             = (indices_shift[1] > 0) ? :(($_ix + $(indices_shift[1]))) : _ix
        iy             = (indices_shift[2] > 0) ? :(($_iy + $(indices_shift[2]))) : _iy
        iz             = (indices_shift[3] > 0) ? :(($_iz + $(indices_shift[3]))) : _iz
        i              = (indices_shift[3] > 0) ? :(($_i  - $(indices_shift[3]))) : _i
        range_z        = (indices_shift[3] > 0) ? :(($ranges[3])[$tz_g] - $(indices_shift[3])) : :(($ranges[3])[$tz_g])
        tx             = gensym_world("tx", @__MODULE__)
        ty             = gensym_world("ty", @__MODULE__)
        ix_h           = gensym_world("ix_h", @__MODULE__)
        ix_h2          = gensym_world("ix_h2", @__MODULE__)
        iy_h           = gensym_world("iy_h", @__MODULE__)
        iy_h2          = gensym_world("iy_h2", @__MODULE__)
        loopoffset     = gensym_world("loopoffset", @__MODULE__)
        A_izp1         = gensym_world(string(A, "_izp1"), @__MODULE__)
        A_ix_iy_izm1   = gensym_world(string(A, "_ix_iy_izm1"), @__MODULE__)
        A_ix_iy_iz     = gensym_world(string(A, "_ix_iy_iz"), @__MODULE__)
        A_ix_iy_izp1   = gensym_world(string(A, "_ix_iy_izp1"), @__MODULE__)
        A_ixm1_iy_iz   = gensym_world(string(A, "_ixm1_iy_iz"), @__MODULE__)
        A_ixp1_iy_iz   = gensym_world(string(A, "_ixp1_iy_iz"), @__MODULE__)
        A_ix_iym1_iz   = gensym_world(string(A, "_ix_iym1_iz"), @__MODULE__)
        A_ix_iyp1_iz   = gensym_world(string(A, "_ix_iyp1_iz"), @__MODULE__)

        body = substitute(body, :($A[$ix,$iy,$iz-1]), A_ix_iy_izm1)
        body = substitute(body, :($A[$ix,$iy,$iz  ]), A_ix_iy_iz  )
        body = substitute(body, :($A[$ix,$iy,$iz+1]), A_ix_iy_izp1)
        if hx > 0
            body = substitute(body, :($A[$ix-1,$iy,$iz]), A_ixm1_iy_iz)
            body = substitute(body, :($A[$ix+1,$iy,$iz]), A_ixp1_iy_iz)
        end
        if hy > 0
            body = substitute(body, :($A[$ix,$iy-1,$iz]), A_ix_iym1_iz)
            body = substitute(body, :($A[$ix,$iy+1,$iz]), A_ix_iyp1_iz)
        end
        if indices_shift[3] > 0
            body =  quote 
                        if ($i > 0)
                            $body
                        end
                    end
        end

        return quote
                    $tx            = @threadIdx().x + $hx
                    $ty            = @threadIdx().y + $hy
                    $ix_h          = ParallelStencil.@ix_h($hx)
                    $ix_h2         = ParallelStencil.@ix_h2($hx, $hy)
                    $iy_h          = ParallelStencil.@iy_h($hx, $hy)
                    $iy_h2         = ParallelStencil.@iy_h2($hx, $hy)
                    $loopoffset    = (@blockIdx().z-1)*$loopsize #TODO: MOVE UP - see no perf change! interchange other lines!
$(shmem ? :(        $A_izp1        = @sharedMem(eltype($A), (ParallelStencil.@nx_l($hx), ParallelStencil.@ny_l($hy)))    ) : noexpr)
                    $A_ix_iy_izm1  = 0.0
                    $A_ix_iy_iz    = $A[$ix,$iy,1+$loopoffset]
                    $A_ix_iy_izp1  = 0.0
$(hx>0 ?  :(        $A_ixm1_iy_iz  = 0.0                                                 ) : noexpr)
$(hx>0 ?  :(        $A_ixp1_iy_iz  = 0.0                                                 ) : noexpr)
$(hy>0 ?  :(        $A_ix_iym1_iz  = 0.0                                                 ) : noexpr)
$(hy>0 ?  :(        $A_ix_iyp1_iz  = 0.0                                                 ) : noexpr)
                    for $_i = 1:$loopsize
                        $tz_g = $_i + $loopoffset
                        if ($tz_g > $rangelength_z) return; end
                        $_iz = $range_z
$(shmem ? quote           
                        @sync_threads()
                        if (ParallelStencil.@t_h() <= cld(ParallelStencil.@nx_l($hx)*ParallelStencil.@ny_l($hy),2) && $ix_h>0 && $ix_h<=size($A,1) && $iy_h>0 && $iy_h<=size($A,2) && $iz<size($A,3)) 
                            $A_izp1[ParallelStencil.@tx_h($hx),ParallelStencil.@ty_h($hx)] = $A[$ix_h,$iy_h,$iz+1] 
                        end
                        if (ParallelStencil.@t_h2($hx, $hy) > cld(ParallelStencil.@nx_l($hx)*ParallelStencil.@ny_l($hy),2) && $ix_h2>0 && $ix_h2<=size($A,1) && $iy_h2>0 && $iy_h2<=size($A,2) && $iz<size($A,3)) 
                            $A_izp1[ParallelStencil.@tx_h2($hx, $hy),ParallelStencil.@ty_h2($hx, $hy)] = $A[$ix_h2,$iy_h2,$iz+1]
                        end
                        @sync_threads()
          end : 
          noexpr
)
$(shmem ? :(            $A_ix_iy_izp1 = $A_izp1[$tx,$ty]
           ) :
          :(            $A_ix_iy_izp1 = ($iz<size($A,3)) ? $A[$ix,$iy,$iz+1] : $A_ix_iy_izp1
           )
)
                        $body
$(hx>0 ? :(             $A_ixm1_iy_iz = $A_izp1[$tx-1,$ty]                                ) : noexpr)
$(hx>0 ? :(             $A_ixp1_iy_iz = $A_izp1[$tx+1,$ty]                                ) : noexpr)
$(hy>0 ? :(             $A_ix_iym1_iz = $A_izp1[$tx,$ty-1]                                ) : noexpr)
$(hy>0 ? :(             $A_ix_iyp1_iz = $A_izp1[$tx,$ty+1]                                ) : noexpr)
                        $A_ix_iy_izm1 = $A_ix_iy_iz
                        $A_ix_iy_iz   = $A_ix_iy_izp1
                    end
        end
    else
        @ArgumentError("@loopopt: only optdim=3 is currently supported.")
    end
end


function loopopt(caller::Module, indices, optvars, optdim::Integer, loopsize, halosize, body; package::Symbol=get_package())
    indices_shift = (0,0,0)
    return loopopt(caller, indices, optvars, optdim, loopsize, halosize, indices_shift, body; package=package)
end


function loopopt(caller::Module, indices, optvars, body; package::Symbol=get_package())
    optdim   = isa(indices,Expr) ? length(indices.args) : 1
    loopsize = LOOPSIZE
    halosize = (optdim == 3) ? (1,1) : (optdim == 2) ? 1 : 0
    return loopopt(caller, indices, optvars, optdim, loopsize, halosize, body; package=package)
end


function shortif(else_val, if_expr; package::Symbol=get_package())
    if (package ∉ SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    @capture(if_expr, if condition_ body_ end) || @ArgumentError("@shortif: the second argument must be an if statement.")
    @capture(body, lhs_ = rhs_) || @ArgumentError("@shortif: the if statement body must contain a assignement.")
    return :($lhs = $condition ? $rhs : $else_val)
end


## FUNCTIONS FOR SHARED MEMORY ALLOCATION

