#TODO: add ParallelStencil.ParallelKernel. in front of all kernel lang in macros! Later: generalize more for z?

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
        rangelength_x  = RANGELENGTHS_VARNAMES[1]
        rangelength_y  = RANGELENGTHS_VARNAMES[2]
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
        range_z_start  = (indices_shift[3] > 0) ? :(($ranges[3])[1]     - $(indices_shift[3])) : :(($ranges[3])[1])
        bx             = gensym_world("bx", @__MODULE__)
        by             = gensym_world("by", @__MODULE__)
        tx             = gensym_world("tx", @__MODULE__)
        ty             = gensym_world("ty", @__MODULE__)
        nx_l           = gensym_world("nx_l", @__MODULE__)
        ny_l           = gensym_world("ny_l", @__MODULE__)
        t_h            = gensym_world("t_h", @__MODULE__)
        t_h2           = gensym_world("t_h2", @__MODULE__)
        tx_h           = gensym_world("tx_h", @__MODULE__)
        tx_h2          = gensym_world("tx_h2", @__MODULE__)
        ty_h           = gensym_world("ty_h", @__MODULE__)
        ty_h2          = gensym_world("ty_h2", @__MODULE__)
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
        else
            body =  quote 
                        if ($_i > 0)
                            $body
                        end
                    end
        end
        loopstart = 0 #TODO: if in z neighbors beyond iz-1 are accessed, then this needs to be bigger (also range_z_start-...!). NOTE: not the same as halosize which is for shmemhalo

        return quote
                    $bx            = (@blockIdx().x*@blockDim().x - $rangelength_x > 0) ? @blockDim().x - (@blockIdx().x*@blockDim().x - $rangelength_x) : @blockDim().x
                    $by            = (@blockIdx().y*@blockDim().y - $rangelength_y > 0) ? @blockDim().y - (@blockIdx().y*@blockDim().y - $rangelength_y) : @blockDim().y
                    $tx            = @threadIdx().x + $hx
                    $ty            = @threadIdx().y + $hy
                    $nx_l          = $bx + (2*$hx)
                    $ny_l          = $by + (2*$hy)
                    $t_h           = (@threadIdx().y-1)*$bx + @threadIdx().x # NOTE: here it must be bx, not @blockDim().x
                    $t_h2          = $t_h + $nx_l*$ny_l - $bx*$by
                    $tx_h          = ($t_h-1) % $nx_l + 1
                    $ty_h          = ($t_h-1) ÷ $nx_l + 1
                    $tx_h2         = ($t_h2-1) % $nx_l + 1
                    $ty_h2         = ($t_h2-1) ÷ $nx_l + 1
                    $ix_h          = (@blockIdx().x-1)*@blockDim().x + $tx_h  - $hx    # NOTE: here it must be @blockDim().x, not bx
                    $ix_h2         = (@blockIdx().x-1)*@blockDim().x + $tx_h2 - $hx    # ...
                    $iy_h          = (@blockIdx().y-1)*@blockDim().y + $ty_h  - $hy    # ...
                    $iy_h2         = (@blockIdx().y-1)*@blockDim().y + $ty_h2 - $hy    # ...
                    $loopoffset    = (@blockIdx().z-1)*$loopsize #TODO: MOVE UP - see no perf change! interchange other lines!
$(shmem ? :(        $A_izp1        = @sharedMem(eltype($A), ($nx_l, $ny_l))              ) : noexpr)
                    $A_ix_iy_izm1  = 0.0
                    $A_ix_iy_iz    = (@blockIdx().z>1) ? $A[$ix,$iy,$range_z_start-1+$loopoffset] : 0.0
                    $A_ix_iy_izp1  = 0.0
$(hx>0 ?  :(        $A_ixm1_iy_iz  = 0.0                                                 ) : noexpr)
$(hx>0 ?  :(        $A_ixp1_iy_iz  = 0.0                                                 ) : noexpr)
$(hy>0 ?  :(        $A_ix_iym1_iz  = 0.0                                                 ) : noexpr)
$(hy>0 ?  :(        $A_ix_iyp1_iz  = 0.0                                                 ) : noexpr)

                    for $_i = $loopstart:$loopsize
                        $tz_g = $_i + $loopoffset
                        if ($tz_g > $rangelength_z) return; end
                        $_iz = ($tz_g < 1) ? $range_z_start-1 : $range_z
$(shmem ? quote           
                        @sync_threads()
                        if ($t_h <= cld($nx_l*$ny_l,2) && $ix_h>0 && $ix_h<=size($A,1) && $iy_h>0 && $iy_h<=size($A,2) && $iz<size($A,3)) 
                            $A_izp1[$tx_h,$ty_h] = $A[$ix_h,$iy_h,$iz+1] 
                        end
                        if ($t_h2 > cld($nx_l*$ny_l,2) && $ix_h2>0 && $ix_h2<=size($A,1) && $iy_h2>0 && $iy_h2<=size($A,2) && $iz<size($A,3)) 
                            $A_izp1[$tx_h2,$ty_h2] = $A[$ix_h2,$iy_h2,$iz+1]
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

