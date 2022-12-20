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


#function add_loopopt(body::Expr, indices::Array{<:Union{Expr,Symbol}}, optvars::Union{Expr,Symbol}, optdim::Integer, loopsize::Union{Expr,Symbol,Integer}, stencilranges::Union{Integer,NTuple{N,Integer} where N})
#TODO: see what to do with global consts as SHMEM_HALO_X,... Support later multiple vars for opt (now just A=T...)
#TODO: add input check and errors
#TODO: maybe gensym with macro @gensym
# TODO: create a run time check for requirement: 
# In order to be able to read the data into shared memory in only two statements, the number of threats must be at least half of the size of the shared memory block plus halo; thus, the total number of threads in each dimension must equal the range length, as else there would be smaller thread blocks at the boundaries (threads overlapping the range are sent home). These smaller blocks would be likely not to match the criteria for a correct reading of the data to shared memory. In summary the following requirements must be matched: @gridDim().x*@blockDim().x - $rangelength_x == 0; @gridDim().y*@blockDim().y - $rangelength_y > 0
function loopopt(caller::Module, indices, optvars, optdim::Integer, loopsize, stencilranges, indices_shift, body; package::Symbol=get_package())
    if !isa(optvars, Symbol) @KeywordArgumentError("at present, only one optvar is supported.") end
    A = optvars 
    if (package ∉ SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    if isa(indices,Expr) indices = indices.args else indices = (indices,) end
    stencilranges = eval_arg(caller, stencilranges)
    rx, ry ,rz    = stencilranges
    rx1, ry1, rz1 = rx.start, ry.start ,rz.start
    rx2, ry2, rz2 = rx.stop, ry.stop ,rz.stop
    noexpr = :(begin end)
    if optdim == 3
        ranges         = RANGES_VARNAME
        rangelength_z  = RANGELENGTHS_VARNAMES[3]
        tz_g           = THREADIDS_VARNAMES[3]
        hx1,hx2, hy1,hy2 = -rx1,rx2, -ry1,ry2
        shmem          = (hx1+hx2>0 || hy1+hy2>0)
        _ix, _iy, _iz  = indices
        _i             = gensym_world("i", @__MODULE__)
        ix             = _ix # (indices_shift[1] > 0) ? :(($_ix + $(indices_shift[1]))) : _ix
        iy             = _iy # (indices_shift[2] > 0) ? :(($_iy + $(indices_shift[2]))) : _iy
        iz             = _iz # (indices_shift[3] > 0) ? :(($_iz + $(indices_shift[3]))) : _iz
        i              = _i # (indices_shift[3] > 0) ? :(($_i  - $(indices_shift[3]))) : _i
        range_z        = :(($ranges[3])[$tz_g]) # (indices_shift[3] > 0) ? :(($ranges[3])[$tz_g] - $(indices_shift[3])) : :(($ranges[3])[$tz_g])
        range_z_start  = :(($ranges[3])[1])     # (indices_shift[3] > 0) ? :(($ranges[3])[1]     - $(indices_shift[3])) : :(($ranges[3])[1])
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

        # 1. Do a loop over the stencil sizes to create all possible indices within stensil; try to substitute the corresponding expressions
        # with the on the fly created symbol. Substitute must return the number of substituted elements.
        # If the number is greater than one, then create the symbol, allocate it later and populate it within the snake.
        # Even if the symbol is not put into the body, it will need to be created and allocated if it is within the snake.
        # The snake is field as follows: before the body is computed anything from the z plus ? plane is red; the rest is read afterwards.
        # Stencil sizes can also be negative, indicating that the element at point zero is never accessed; this avoids unnecessary reads 
        # and unnecessary shared memory allocation. The snake does not need to go until point zero if it is not needed.
        # UPDATE: stencil sizes should be called stencil ranges instead.
        if (rz1 < 0) body = substitute(body, :($A[$ix,$iy,$iz-1]), A_ix_iy_izm1) end
        body = substitute(body, :($A[$ix,$iy,$iz  ]), A_ix_iy_iz  )
        if (rz2 > 0) body = substitute(body, :($A[$ix,$iy,$iz+1]), A_ix_iy_izp1) end
        # if hx > 0
            if (rx1 < 0) body = substitute(body, :($A[$ix-1,$iy,$iz]), A_ixm1_iy_iz) end
            if (rx2 > 0) body = substitute(body, :($A[$ix+1,$iy,$iz]), A_ixp1_iy_iz) end
        # end
        # if hy > 0
            if (ry1 < 0) body = substitute(body, :($A[$ix,$iy-1,$iz]), A_ix_iym1_iz) end
            if (ry1 > 0) body = substitute(body, :($A[$ix,$iy+1,$iz]), A_ix_iyp1_iz) end
        # end
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
        loopstart = 0 #TODO: if in z neighbors beyond iz-1 are accessed, then this needs to be bigger (also range_z_start-...!). NOTE: not the same as stencilranges which is for shmemhalo

        return quote
                    $tx            = @threadIdx().x + $hx1
                    $ty            = @threadIdx().y + $hy1
$(shmem ? quote     
                    $nx_l          = @blockDim().x + $(hx1+hx2)
                    $ny_l          = @blockDim().y + $(hy1+hy2)
                    $t_h           = (@threadIdx().y-1)*@blockDim().x + @threadIdx().x # NOTE: here it must be bx, not @blockDim().x
                    $t_h2          = $t_h + $nx_l*$ny_l - @blockDim().x*@blockDim().y
                    $tx_h          = ($t_h-1) % $nx_l + 1
                    $ty_h          = ($t_h-1) ÷ $nx_l + 1
                    $tx_h2         = ($t_h2-1) % $nx_l + 1
                    $ty_h2         = ($t_h2-1) ÷ $nx_l + 1
                    $ix_h          = (@blockIdx().x-1)*@blockDim().x + $tx_h  - $hx1    # NOTE: here it must be @blockDim().x, not bx
                    $ix_h2         = (@blockIdx().x-1)*@blockDim().x + $tx_h2 - $hx1    # ...
                    $iy_h          = (@blockIdx().y-1)*@blockDim().y + $ty_h  - $hy1    # ...
                    $iy_h2         = (@blockIdx().y-1)*@blockDim().y + $ty_h2 - $hy1    # ...
        end : 
        noexpr
)
                    $loopoffset    = (@blockIdx().z-1)*$loopsize #TODO: MOVE UP - see no perf change! interchange other lines!
$(shmem ? :(        $A_izp1        = @sharedMem(eltype($A), ($nx_l, $ny_l))              ) : noexpr)
                    $A_ix_iy_izm1  = 0.0
                    $A_ix_iy_iz    = (@blockIdx().z>1) ? $A[$ix,$iy,$range_z_start-1+$loopoffset] : 0.0
                    $A_ix_iy_izp1  = 0.0
$(rx1<0 ?  :(       $A_ixm1_iy_iz  = 0.0                                                 ) : noexpr)
$(rx2>0 ?  :(       $A_ixp1_iy_iz  = 0.0                                                 ) : noexpr)
$(ry1<0 ?  :(       $A_ix_iym1_iz  = 0.0                                                 ) : noexpr)
$(ry2>0 ?  :(       $A_ix_iyp1_iz  = 0.0                                                 ) : noexpr)

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
$(rx1<0 ? :(            $A_ixm1_iy_iz = $A_izp1[$tx-1,$ty]                                ) : noexpr)
$(rx2>0 ? :(            $A_ixp1_iy_iz = $A_izp1[$tx+1,$ty]                                ) : noexpr)
$(ry1<0 ? :(            $A_ix_iym1_iz = $A_izp1[$tx,$ty-1]                                ) : noexpr)
$(ry2>0 ? :(            $A_ix_iyp1_iz = $A_izp1[$tx,$ty+1]                                ) : noexpr)
                        $A_ix_iy_izm1 = $A_ix_iy_iz
                        $A_ix_iy_iz   = $A_ix_iy_izp1
                    end
        end
    else
        @ArgumentError("@loopopt: only optdim=3 is currently supported.")
    end
end


function loopopt(caller::Module, indices, optvars, optdim::Integer, loopsize, stencilranges, body; package::Symbol=get_package())
    indices_shift = (0,0,0)
    return loopopt(caller, indices, optvars, optdim, loopsize, stencilranges, indices_shift, body; package=package)
end


function loopopt(caller::Module, indices, optvars, body; package::Symbol=get_package())
    optdim   = isa(indices,Expr) ? length(indices.args) : 1
    loopsize = LOOPSIZE
    stencilranges = (optdim == 3) ? (-1:1,-1:1,-1:1) : (optdim == 2) ? (-1:1,-1:1,0:0) : (-1:1,0:0,0:0)
    return loopopt(caller, indices, optvars, optdim, loopsize, stencilranges, body; package=package)
end


function shortif(else_val, if_expr; package::Symbol=get_package())
    if (package ∉ SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    @capture(if_expr, if condition_ body_ end) || @ArgumentError("@shortif: the second argument must be an if statement.")
    @capture(body, lhs_ = rhs_) || @ArgumentError("@shortif: the if statement body must contain a assignement.")
    return :($lhs = $condition ? $rhs : $else_val)
end


## FUNCTIONS FOR SHARED MEMORY ALLOCATION

