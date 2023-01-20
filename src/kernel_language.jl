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
function loopopt(caller::Module, indices, optvars, optdim::Integer, loopsize, stencilranges, body; package::Symbol=get_package())
    if !isa(optvars, Symbol) @KeywordArgumentError("at present, only one optvar is supported.") end
    A = optvars 
    if (package ∉ SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    if     (package == PKG_CUDA)    int_type = INT_CUDA
    elseif (package == PKG_AMDGPU)  int_type = INT_AMDGPU
    elseif (package == PKG_THREADS) int_type = INT_THREADS
    end
    if isa(indices,Expr) indices = indices.args else indices = (indices,) end
    fullrange     = typemin(int_type):typemax(int_type)
    stencilranges = isnothing(stencilranges) ? (fullrange, fullrange, fullrange) : eval_arg(caller, stencilranges)
    body          = eval_offsets(caller, body, indices, int_type)
    offsets       = extract_offsets(caller, body, indices, int_type, optdim)
    regqueue_head, regqueue_tail, offset_min, offset_max = define_regqueue(offsets, stencilranges, A, indices, int_type, optdim)
    @show regqueue_head, regqueue_tail, offset_min, offset_max
    if optdim == 3
        ranges         = RANGES_VARNAME
        rangelength_z  = RANGELENGTHS_VARNAMES[3]
        tz_g           = THREADIDS_VARNAMES[3]
        oz_max         = offset_max[3]
        hx1, hy1       = -1 .* offset_min[1:2]
        hx2, hy2       = offset_max[1:2]
        shmem          = (hx1+hx2>0 || hy1+hy2>0)
        offset_span    = offset_max .- offset_min
        oz_span        = offset_span[3]
        loopstart      = 1 - oz_span #TODO: make possibility to do first and last read in z dimension directly into registers without halo
        ix, iy, iz     = indices
        i              = gensym_world("i", @__MODULE__)
        range_z        = :(($ranges[3])[$tz_g])
        range_z_start  = :(($ranges[3])[1])
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
        A_head         = gensym_world(varname(A, (oz_max,); i="iz"), @__MODULE__)

        for oxy in keys(regqueue_tail)
            for oz in keys(regqueue_tail[oxy])
                body = substitute(body, regtarget(A, (oxy..., oz), indices), regqueue_tail[oxy][oz])
            end
        end
        for oxy in keys(regqueue_head)
            for oz in keys(regqueue_head[oxy])
                body = substitute(body, regtarget(A, (oxy..., oz), indices), regqueue_head[oxy][oz])
            end
        end
        body =  quote 
                    if ($i > 0)
                        $body
                    end
                end

        return quote
$(shmem ? quote     
                    $tx            = @threadIdx().x + $hx1
                    $ty            = @threadIdx().y + $hy1
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
        NOEXPR
)
                    $loopoffset    = (@blockIdx().z-1)*$loopsize #TODO: MOVE UP - see no perf change! interchange other lines!
$(shmem ? :(        $A_head        = @sharedMem(eltype($A), ($nx_l, $ny_l))              ) : NOEXPR) # e.g. A_izp3 = @sharedMem(eltype(A), (nx_l, ny_l))
$((:(               $reg           = 0.0                                                # e.g. A_ixm1_iyp2_izp2 = 0.0
    ) 
    for regs in values(regqueue_tail) for reg in values(regs)
  )...
)
$((:(               $reg           = 0.0                                                # e.g. A_ixm1_iyp2_izp3 = 0.0
    )
    for regs in values(regqueue_head) for reg in values(regs)
  )...
)
                    for $i = $loopstart:$loopsize
                        $tz_g = $i + $loopoffset
                        if ($tz_g > $rangelength_z) return; end
                        $iz = ($tz_g < 1) ? $range_z_start-(1-$tz_g) : $range_z # TODO: this will probably always be formulated with range_z_start
$(shmem ? quote           
                        @sync_threads()
                        if ($t_h <= cld($nx_l*$ny_l,2) && $ix_h>0 && $ix_h<=size($A,1) && $iy_h>0 && $iy_h<=size($A,2) && 0<$iz+$oz_max<=size($A,3)) 
                            $A_head[$tx_h,$ty_h] = $A[$ix_h,$iy_h,$iz+$oz_max] 
                        end
                        if ($t_h2 > cld($nx_l*$ny_l,2) && $ix_h2>0 && $ix_h2<=size($A,1) && $iy_h2>0 && $iy_h2<=size($A,2) && 0<$iz+$oz_max<=size($A,3)) 
                            $A_head[$tx_h2,$ty_h2] = $A[$ix_h2,$iy_h2,$iz+$oz_max]
                        end
                        @sync_threads()
          end : 
          NOEXPR
)
$((shmem ?
  (:(                   $reg       = $(regsource(A_head, oxy, (tx, ty)))                # e.g. A_ixm1_iyp2_izp3 = A_izp3[tx - 1, ty + 2]
    )
    for (oxy, regs) in regqueue_head for reg in values(regs)
  ) :
  (:(                   $reg       = (0<$ix+$(oxy[1])<=size($A,1) && 0<$iy+$(oxy[2])<=size($A,2) && 0<$iz+$oz<=size($A,3)) ? $(regtarget(A, (oxy...,oz), indices)) : $reg
    )
    for (oxy, regs) in regqueue_head for (oz, reg) in regs
  ))...
)

                        $body

$((:(
                        $(regs[oz]) = $(regs[oz+1])                                     # e.g. A_ixm1_iyp2_iz = A_ixm1_iyp2_izp1
    )
    for regs in values(regqueue_tail) for oz in sort(keys(regs)) if oz<=oz_max-2
  )...
)
$((:(                   $reg        = $(regsource(A_head, oxy, (tx, ty)))               # e.g. A_ixm3_iyp2_izp2 = A_izp3[tx - 3, ty + 2]
    )
    for (oxy, regs) in regqueue_tail for (oz, reg) in regs if oz==oz_max-1 && !(haskey(regqueue_head, oxy) && haskey(regqueue_head[oxy], oz_max))
  )...
)
$((:(
                        $reg           = $(regqueue_head[oxy][oz_max])                  # e.g. A_ixm1_iyp2_izp2 = A_ixm1_iyp2_izp3
    )
    for (oxy, regs) in regqueue_tail for (oz, reg) in regs if oz==oz_max-1 && haskey(regqueue_head, oxy) && haskey(regqueue_head[oxy], oz_max)
  )...
)
                    end
        end
    else
        @ArgumentError("@loopopt: only optdim=3 is currently supported.")
    end
end


function loopopt(caller::Module, indices, optvars, body; package::Symbol=get_package())
    optdim        = isa(indices,Expr) ? length(indices.args) : 1
    loopsize      = LOOPSIZE
    stencilranges = nothing
    return loopopt(caller, indices, optvars, optdim, loopsize, stencilranges, body; package=package)
end


function shortif(else_val, if_expr; package::Symbol=get_package())
    if (package ∉ SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    @capture(if_expr, if condition_ body_ end) || @ArgumentError("@shortif: the second argument must be an if statement.")
    @capture(body, lhs_ = rhs_) || @ArgumentError("@shortif: the if statement body must contain a assignement.")
    return :($lhs = $condition ? $rhs : $else_val)
end


## FUNCTIONS FOR SHARED MEMORY ALLOCATION


## HELPER FUNCTIONS

is_stencil_access(ex::Expr, ix::Symbol, iy::Symbol, iz::Symbol) = @capture(ex, A_[x_, y_, z_]) && inexpr_walk(x, ix) && inexpr_walk(y, iy) && inexpr_walk(z, iz)
is_stencil_access(ex::Expr, ix::Symbol, iy::Symbol)             = @capture(ex, A_[x_, y_])     && inexpr_walk(x, ix) && inexpr_walk(y, iy)
is_stencil_access(ex::Expr, ix::Symbol)                         = @capture(ex, A_[x_])         && inexpr_walk(x, ix)
is_stencil_access(ex, indices...)                               = false

function eval_offsets(caller::Module, body, indices, int_type)
    return postwalk(body) do ex
        if !is_stencil_access(ex, indices...) return ex; end
        @capture(ex, A_[indices_expr__]) || @ModuleInternalError("a stencil access could not be pattern matched.")
        for i = 1:length(indices)
            offset_expr = substitute(indices_expr[i], indices[i], 0)
            offset = int_type(eval_arg(caller, offset_expr))
            if     (offset >  0) indices_expr[i] = :($(indices[i]) + $offset       )
            elseif (offset <  0) indices_expr[i] = :($(indices[i]) - $(abs(offset)))
            else                 indices_expr[i] =     indices[i]
            end
        end
        return :($A[$(indices_expr...)])
    end
end

function extract_offsets(caller::Module, body, indices, int_type, optdim)
    access_offsets = Dict()
    postwalk(body) do ex
        if is_stencil_access(ex, indices...)
            @capture(ex, A_[indices_expr__]) || @ModuleInternalError("a stencil access could not be pattern matched.")
            offsets = ()
            for i = 1:length(indices)
                offset_expr = substitute(indices_expr[i], indices[i], 0)
                offset = int_type(eval_arg(caller, offset_expr))
                offsets = (offsets..., offset)
            end
            if optdim == 3
                k1 = offsets[1:2]
                k2 = offsets[end]
                if     haskey(access_offsets, k1) && haskey(access_offsets[k1], k2) access_offsets[k1][k2] += 1
                elseif haskey(access_offsets, k1)                                   access_offsets[k1][k2]  = 1
                else                                                                access_offsets[k1]      = Dict(k2 => 1)
                end
            else
                @ArgumentError("@loopopt: only optdim=3 is currently supported.")
            end
        end
        return ex    
    end
    return access_offsets
end

function define_regqueue(offsets, stencilranges, A, indices, int_type, optdim)
    regqueue_head = Dict()
    regqueue_tail = Dict()
    if optdim == 3
        stencilranges_xy = stencilranges[1:2]
        stencilranges_z  = stencilranges[3]
        offsets_xy       = filter(oxy -> all(oxy .∈ stencilranges_xy), keys(offsets))
        if isempty(offsets_xy) @IncoherentArgumentError("incoherent argument in @loopopt: stencilranges in x-y dimension do not include any array access.") end
        offset_min       = (typemax(int_type), typemax(int_type), typemax(int_type))
        offset_max       = (typemin(int_type), typemin(int_type), typemin(int_type))
        for oxy in offsets_xy
            offsets_z = filter(x -> x ∈ stencilranges_z, keys(offsets[oxy]))
            if isempty(offsets_z) @IncoherentArgumentError("incoherent argument in @loopopt: stencilranges in z dimension do not include any array access.") end
            offset_min = (min(offset_min[1], oxy[1]),
                          min(offset_min[2], oxy[2]),
                          min(offset_min[3], minimum(offsets_z)))
            offset_max = (max(offset_max[1], oxy[1]),
                          max(offset_max[2], oxy[2]),
                          max(offset_max[3], maximum(offsets_z)))
        end
        oz_max = offset_max[3]
        for oxy in offsets_xy
            offsets_z = sort(filter(x -> x ∈ stencilranges_z, keys(offsets[oxy])))
            k1 = oxy
            for oz = offsets_z[1]:oz_max-1
                k2 = oz
                if haskey(regqueue_tail, k1) && haskey(regqueue_tail[k1], k2) @ModuleInternalError("regqueue_tail entry exists already.") end
                reg = gensym_world(varname(A, (oxy..., oz)), @__MODULE__)
                if haskey(regqueue_tail, k1) regqueue_tail[k1][k2] = reg
                else                         regqueue_tail[k1]     = Dict(k2 => reg)
                end
            end
            oz = offsets_z[end]
            if oz == oz_max
                k2 = oz
                if haskey(regqueue_head, k1) && haskey(regqueue_head[k1], k2) @ModuleInternalError("regqueue_head entry exists already.") end
                reg = gensym_world(varname(A, (oxy..., oz)), @__MODULE__)
                if haskey(regqueue_head, k1) regqueue_head[k1][k2] = reg
                else                         regqueue_head[k1]     = Dict(k2 => reg)
                end
            end
        end
    else
        @ArgumentError("@loopopt: only optdim=3 is currently supported.")
    end
    return regqueue_head, regqueue_tail, offset_min, offset_max
end

function varname(A, offsets; i="ix", j="iy", k="iz")
    ndims = length(offsets)
    ox    = offsets[1]
    x = if     (ox > 0) i * "p" * string(ox)
        elseif (ox < 0) i * "m" * string(abs(ox))
        else            i
        end
    if ndims > 1
        oy = offsets[2]
        y = if     (oy > 0) j * "p" * string(oy)
            elseif (oy < 0) j * "m" * string(abs(oy))
            else            j
            end
    end
    if ndims > 2
        oz = offsets[3]
        z = if     (oz > 0) k * "p" * string(oz)
            elseif (oz < 0) k * "m" * string(abs(oz))
            else            k
            end
    end
    if     (ndims == 1) return string(A, "_$(x)")
    elseif (ndims == 2) return string(A, "_$(x)_$(y)")
    elseif (ndims == 3) return string(A, "_$(x)_$(y)_$(z)")
    end
end

function regtarget(A, offsets, indices)
    ndims = length(offsets)
    ox    = offsets[1]
    ix    = indices[1]
    if     (ox > 0) x = :($ix + $ox)
    elseif (ox < 0) x = :($ix - $(abs(ox)))
    else            x = ix
    end
    if ndims > 1
        oy = offsets[2]
        iy = indices[2]
        if     (oy > 0) y = :($iy + $oy)
        elseif (oy < 0) y = :($iy - $(abs(oy)))
        else            y = iy
        end
    end
    if ndims > 2
        oz = offsets[3]
        iz = indices[3]
        if     (oz > 0) z = :($iz + $oz)
        elseif (oz < 0) z = :($iz - $(abs(oz)))
        else            z = iz
        end
    end
    if     (ndims == 1) return :($A[$x])
    elseif (ndims == 2) return :($A[$x,$y])
    elseif (ndims == 3) return :($A[$x,$y,$z])
    end
end

function regsource(A_head, offsets, local_indices)
    ndims = length(offsets)
    ox    = offsets[1]
    tx    = local_indices[1]
    if     (ox > 0) x = :($tx + $ox)
    elseif (ox < 0) x = :($tx - $(abs(ox)))
    else            x = tx
    end
    if ndims > 1
        oy    = offsets[2]
        ty    = local_indices[2]
        if     (oy > 0) y = :($ty + $oy)
        elseif (oy < 0) y = :($ty - $(abs(oy)))
        else            y = ty
        end
    end
    if     (ndims == 1) return :($A_head[$x])
    elseif (ndims == 2) return :($A_head[$x,$y]) # e.g. :($A_head[$tx,$ty-1])
    end
end

Base.sort(keys::T) where T<:Base.AbstractSet = sort([keys...])
