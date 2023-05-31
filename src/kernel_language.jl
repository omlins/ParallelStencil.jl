#TODO: add ParallelStencil.ParallelKernel. in front of all kernel lang in macros! Later: generalize more for z?

##
macro loop(args...) check_initialized(); checkargs_loop(args...); esc(loop(args...)); end


##
macro memopt(args...) check_initialized(); checkargs_memopt(args...); esc(memopt(args[1], __module__, args[2:end]...)); end


##
macro shortif(args...) check_initialized(); checktwoargs(args...); esc(shortif(args...)); end


##
macro return_nothing(args...) check_initialized(); checknoargs(args...); esc(return_nothing(args...)); end


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

function checkargs_memopt(args...)
    if (length(args) != 8 && length(args) != 7 && length(args) != 4) @ArgumentError("wrong number of arguments.") end
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

#TODO: add input check and errors
# TODO: create a run time check for requirement: 
# In order to be able to read the data into shared memory in only two statements, the number of threats must be at least half of the size of the shared memory block plus halo; thus, the total number of threads in each dimension must equal the range length, as else there would be smaller thread blocks at the boundaries (threads overlapping the range are sent home). These smaller blocks would be likely not to match the criteria for a correct reading of the data to shared memory. In summary the following requirements must be matched: @gridDim().x*@blockDim().x - $rangelength_x == 0; @gridDim().y*@blockDim().y - $rangelength_y > 0
function memopt(metadata_module::Module, is_parallel_kernel::Bool, caller::Module, indices::Union{Symbol,Expr}, optvars::Union{Expr,Symbol}, optdim::Integer, loopsize::Integer, optranges::Union{Nothing, NamedTuple{t, <:NTuple{N,NTuple{3,UnitRange}} where N} where t}, use_shmemhalos::Union{Nothing, NamedTuple{t, <:NTuple{N,Bool} where N} where t}, optimize_halo_read::Bool, body::Expr; package::Symbol=get_package())
    optvars        = Tuple(extract_tuple(optvars)) #TODO: make this function actually return directly a tuple rather than an array
    indices        = Tuple(extract_tuple(indices))
    use_shmemhalos = isnothing(use_shmemhalos) ? use_shmemhalos : eval_arg(caller, use_shmemhalos)
    optranges      = isnothing(optranges) ? optranges : eval_arg(caller, optranges)
    readonlyvars   = find_readonlyvars(body, indices)
    if optvars == (Symbol(""),)
        optvars = Tuple(keys(readonlyvars))
    else
        for A in optvars
            if !haskey(readonlyvars, A) @IncoherentArgumentError("incoherent argument optvars in memopt: optimization can only be applied to arrays that are only read within the kernel (not applicable to: $A).") end
        end
    end
    if (package ∉ SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    if     (package == PKG_CUDA)    int_type = INT_CUDA
    elseif (package == PKG_AMDGPU)  int_type = INT_AMDGPU
    elseif (package == PKG_THREADS) int_type = INT_THREADS
    end
    body                  = eval_offsets(caller, body, indices, int_type)
    offsets, offsets_by_z = extract_offsets(caller, body, indices, int_type, optvars, optdim)
    optvars               = remove_single_point_optvars(optvars, optranges, offsets, offsets_by_z)
    if (length(optvars)==0) @IncoherentArgumentError("incoherent argument memopt in @parallel[_indices] <kernel>: optimization can only be applied if there is at least one array that is read-only within the kernel (and accessed with a multi-point stencil). Set memopt=false for this kernel.") end
    optranges             = define_optranges(optranges, optvars, offsets, int_type)
    regqueue_heads, regqueue_tails, offset_mins, offset_maxs, nb_regs_heads, nb_regs_tails = define_regqueues(offsets, optranges, optvars, indices, int_type, optdim)

    if optdim == 3
        oz_maxs, hx1s, hy1s, hx2s, hy2s, use_shmems, use_shmem_xs, use_shmem_ys, use_shmemhalos, use_shmemindices, offset_spans, oz_spans, loopentrys = define_helper_variables(offset_mins, offset_maxs, optvars, use_shmemhalos, optdim)
        loopstart          = minimum(values(loopentrys))
        loopend            = loopsize
        use_any_shmem      = any(values(use_shmems))
        shmem_index_groups = define_shmem_index_groups(hx1s, hy1s, hx2s, hy2s, optvars, use_shmems, optdim)
        shmem_vars         = define_shmem_vars(oz_maxs, hx1s, hy1s, hx2s, hy2s, optvars, indices, use_shmems, use_shmem_xs, use_shmem_ys, shmem_index_groups, use_shmemhalos, use_shmemindices, optdim)
        shmem_exprs        = define_shmem_exprs(shmem_vars, optdim)
        shmem_z_ranges     = define_shmem_z_ranges(offsets_by_z, use_shmems, optdim)
        shmem_loopentrys   = define_shmem_loopentrys(loopentrys, shmem_z_ranges, offset_mins, optdim)
        shmem_loopexits    = define_shmem_loopexits(loopend, shmem_z_ranges, offset_maxs, optdim)
        mainloopstart      = (optimize_halo_read && !isempty(shmem_loopentrys)) ? minimum(values(shmem_loopentrys)) : loopstart
        mainloopend        = loopend # TODO: the second loop split leads to wrong results, probably due to a compiler bug. # mainloopend            = (optimize_halo_read && !isempty(shmem_loopexits) ) ? maximum(values(shmem_loopexits) ) : loopend
        ix, iy, iz         = indices
        tz_g               = THREADIDS_VARNAMES[3]
        rangelength_z      = RANGELENGTHS_VARNAMES[3]
        ranges             = RANGES_VARNAME
        range_z            = :(($ranges[3])[$tz_g])
        range_z_start      = :(($ranges[3])[1])
        i                  = gensym_world("i", @__MODULE__)
        loopoffset         = gensym_world("loopoffset", @__MODULE__)

        for A in optvars
            regqueue_tail = regqueue_tails[A]
            regqueue_head = regqueue_heads[A]
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
        end

        @show nb_indexing_vars = 1 + 14*length(keys(shmem_index_groups)) # TODO: a group must not be counted if none of the variables uses the shmem indices symbols.
        @show nb_cell_vars     = sum(values(nb_regs_heads)) + sum(values(nb_regs_tails))

        #TODO: replace wrap_if where possible with in-line if - compare performance when doing it
        body = quote
                    $loopoffset    = (@blockIdx().z-1)*$loopsize #TODO: MOVE UP - see no perf change! interchange other lines!
$((quote
                    $tx            = @threadIdx().x + $hx1
                    $ty            = @threadIdx().y + $hy1
                    $nx_l          = @blockDim().x + $(hx1+hx2)
                    $ny_l          = @blockDim().y + $(hy1+hy2)
                    $t_h           = (@threadIdx().y-1)*@blockDim().x + @threadIdx().x  # NOTE: here it must be bx, not @blockDim().x
                    $t_h2          = $t_h + $nx_l*$ny_l - @blockDim().x*@blockDim().y
                    $ty_h          = ($t_h-1) ÷ $nx_l + 1
                    $tx_h          = ($t_h-1) % $nx_l + 1                               # NOTE: equivalent to (worse performance has uses registers probably differently): ($t_h-1) - $nx_l*($ty_h-1) + 1
                    $ty_h2         = ($t_h2-1) ÷ $nx_l + 1
                    $tx_h2         = ($t_h2-1) % $nx_l + 1                              # NOTE: equivalent to (worse performance has uses registers probably differently): ($t_h2-1) - $nx_l*($ty_h2-1) + 1
                    $ix_h          = $ix - @threadIdx().x + $tx_h  - $hx1    # NOTE: here it must be @blockDim().x, not bx
                    $ix_h2         = $ix - @threadIdx().x + $tx_h2 - $hx1    # ...
                    $iy_h          = $iy - @threadIdx().y + $ty_h  - $hy1    # ...
                    $iy_h2         = $iy - @threadIdx().y + $ty_h2 - $hy1    # ...
    end
    for vars in values(shmem_index_groups) for A in (vars[1],) if use_shmemindices[A] for s in (shmem_vars[A],) for (shmem_offset,  hx1, hx2, hy1, hy2,  tx, ty, nx_l, ny_l, t_h, t_h2, tx_h, tx_h2, ty_h, ty_h2, ix_h, ix_h2, iy_h, iy_h2, A_head) = ((shmem_exprs[A][:offset],  hx1s[A], hx2s[A], hy1s[A], hy2s[A],  s[:tx], s[:ty], s[:nx_l], s[:ny_l], s[:t_h], s[:t_h2], s[:tx_h], s[:tx_h2], s[:ty_h], s[:ty_h2], s[:ix_h], s[:ix_h2], s[:iy_h], s[:iy_h2], s[:A_head]),)
  )...
)
$((:(               $A_head        = @sharedMem(eltype($A), ($nx_l, $ny_l), $shmem_offset) # e.g. A_izp3 = @sharedMem(eltype(A), (nx_l, ny_l), +(nx_l_A * ny_l_A)*eltype(A))
    )
    for (A, s) in shmem_vars for (shmem_offset,  nx_l, ny_l, A_head) = ((shmem_exprs[A][:offset],  s[:nx_l], s[:ny_l], s[:A_head]),)
  )...
)
$((:(               $reg           = 0.0                                                # e.g. A_ixm1_iyp2_izp2 = 0.0
    ) 
    for A in optvars for regs in values(regqueue_tails[A]) for reg in values(regs)
  )...
)
$((:(               $reg           = 0.0                                                # e.g. A_ixm1_iyp2_izp3 = 0.0
    )
    for A in optvars for regs in values(regqueue_heads[A]) for reg in values(regs)
  )...
)
# Pre-loop
                    for $i = $loopstart:$(mainloopstart-1)
                        $tz_g = $i + $loopoffset
                        if ($tz_g > $rangelength_z) ParallelStencil.@return_nothing; end
                        $iz = ($tz_g < 1) ? $range_z_start-(1-$tz_g) : $range_z # TODO: this will probably always be formulated with range_z_start
$((wrap_if(:($i > $(loopentry-1)),
       :(               $reg       = (0<$ix+$(oxy[1])<=size($A,1) && 0<$iy+$(oxy[2])<=size($A,2) && 0<$iz+$oz<=size($A,3)) ? $(regtarget(A, (oxy...,oz), indices)) : $reg
        )
        ;unless=(loopentry==loopstart)
    )
    for A in keys(shmem_vars) for (oxy, regs) in regqueue_heads[A] for (oz, reg) in regs for loopentry = (loopentrys[A],)
  )...
)
$((wrap_if(:($i > $(loopentry-1)),
       :(               $reg       = (0<$ix+$(oxy[1])<=size($A,1) && 0<$iy+$(oxy[2])<=size($A,2) && 0<$iz+$oz<=size($A,3)) ? $(regtarget(A, (oxy...,oz), indices)) : $reg
        )
        ;unless=(loopentry==loopstart)
    )
    for A in optvars for (oxy, regs) in regqueue_heads[A] for (oz, reg) in regs for loopentry = (loopentrys[A],) if !use_shmems[A]
  )...
)
$(( # NOTE: the if statement is not needed here as we only deal with registers
    # wrap_if(:($i > $(loopentry-1)),
       :(
                        $(regs[oz]) = $(regs[oz+1])                                     # e.g. A_ixm1_iyp2_iz = A_ixm1_iyp2_izp1
        )
        # ;unless=(loopentry==loopstart)
    # )
    for A in optvars for regs in values(regqueue_tails[A]) for oz in sort(keys(regs)) for (loopentry, oz_max) = ((loopentrys[A], oz_maxs[A]),) if oz<=oz_max-2
  )...
)
$(( # NOTE: the if statement is not needed here as we only deal with registers
    # wrap_if(:($i > $(loopentry-1)),
       :(
                        $reg           = $(regqueue_heads[A][oxy][oz_max])              # e.g. A_ixm1_iyp2_izp2 = A_ixm1_iyp2_izp3
        )
        # ;unless=(loopentry==loopstart)
    # )
    for A in optvars for (oxy, regs) in regqueue_tails[A] for (oz, reg) in regs for (loopentry, oz_max) = ((loopentrys[A], oz_maxs[A]),) if oz==oz_max-1 && haskey(regqueue_heads[A], oxy) && haskey(regqueue_heads[A][oxy], oz_max)
  )...
)
                    end

# Main loop
                    # for $i = $mainloopstart:$mainloopend # ParallelStencil.@unroll 
$(wrap_loop(i, mainloopstart:mainloopend, 
        quote
                        $tz_g = $i + $loopoffset
                        if ($tz_g > $rangelength_z) ParallelStencil.@return_nothing; end
                        $iz = ($tz_g < 1) ? $range_z_start-(1-$tz_g) : $range_z # TODO: this will probably always be formulated with range_z_start
$(use_any_shmem ? 
    :(                  @sync_threads()
     ) :                NOEXPR
)
$((wrap_if(:($i > $(loopentry-1)),
        quote
                        if (2*$t_h <= $n_l && $ix_h>0 && $ix_h<=size($A,1) && $iy_h>0 && $iy_h<=size($A,2) && 0<$iz+$oz_max<=size($A,3)) 
                            $A_head[$tx_h,$ty_h] = $A[$ix_h,$iy_h,$iz+$oz_max] 
                        end
                        if (2*$t_h2 > $n_l && $ix_h2>0 && $ix_h2<=size($A,1) && $iy_h2>0 && $iy_h2<=size($A,2) && 0<$iz+$oz_max<=size($A,3)) 
                            $A_head[$tx_h2,$ty_h2] = $A[$ix_h2,$iy_h2,$iz+$oz_max]
                        end
        end
        ;unless=(loopentry<=mainloopstart)
    )
    for (A, s) in shmem_vars if use_shmemhalos[A] for (loopentry, oz_max,  tx, ty, nx_l, ny_l, n_l, t_h, t_h2, tx_h, tx_h2, ty_h, ty_h2, ix_h, ix_h2, iy_h, iy_h2, A_head) = ((loopentrys[A], oz_maxs[A],  s[:tx], s[:ty], s[:nx_l], s[:ny_l], s[:n_l], s[:t_h], s[:t_h2], s[:tx_h], s[:tx_h2], s[:ty_h], s[:ty_h2], s[:ix_h], s[:ix_h2], s[:iy_h], s[:iy_h2], s[:A_head]),)
  )...
)
# $((wrap_if(:($i > $(loopentry-1)),
#         quote
#                         if (2*$tx_h <= $nx_l && $ix_h>0 && $ix_h<=size($A,1) && $iy>0 && $iy<=size($A,2) && 0<$iz+$oz_max<=size($A,3)) 
#                             $A_head[$tx_h,$ty] = $A[$ix_h,$iy,$iz+$oz_max] 
#                         end
#                         if (2*$tx_h2 > $nx_l && $ix_h2>0 && $ix_h2<=size($A,1) && $iy>0 && $iy<=size($A,2) && 0<$iz+$oz_max<=size($A,3)) 
#                             $A_head[$tx_h2,$ty] = $A[$ix_h2,$iy,$iz+$oz_max]
#                         end
#         end
#         ;unless=(loopentry<=mainloopstart)
#     )
#     for (A, s) in shmem_vars if (use_shmemhalos[A] && use_shmem_xs[A] && !use_shmem_ys[A]) for (loopentry, oz_max,  tx, ty, nx_l, ny_l, tx_h, tx_h2, ix_h, ix_h2, A_head) = ((loopentrys[A], oz_maxs[A],  s[:tx], s[:ty], s[:nx_l], s[:ny_l], s[:tx_h], s[:tx_h2], s[:ix_h], s[:ix_h2], s[:A_head]),)
#   )...
# )
# $((wrap_if(:($i > $(loopentry-1)),
#         quote
#                         if (2*$ty_h <= $ny_l && $ix>0 && $ix<=size($A,1) && $iy_h>0 && $iy_h<=size($A,2) && 0<$iz+$oz_max<=size($A,3)) 
#                             $A_head[$tx,$ty_h] = $A[$ix,$iy_h,$iz+$oz_max] 
#                         end
#                         if (2*$ty_h2 > $ny_l && $ix>0 && $ix<=size($A,1) && $iy_h2>0 && $iy_h2<=size($A,2) && 0<$iz+$oz_max<=size($A,3)) 
#                             $A_head[$tx,$ty_h2] = $A[$ix,$iy_h2,$iz+$oz_max]
#                         end
#         end
#         ;unless=(loopentry<=mainloopstart)
#     )
#     for (A, s) in shmem_vars if (use_shmemhalos[A] && !use_shmem_xs[A] && use_shmem_ys[A]) for (loopentry, oz_max,  tx, ty, nx_l, ny_l, ty_h, ty_h2, iy_h, iy_h2, A_head) = ((loopentrys[A], oz_maxs[A],  s[:tx], s[:ty], s[:nx_l], s[:ny_l], s[:ty_h], s[:ty_h2], s[:iy_h], s[:iy_h2], s[:A_head]),)
#   )...
# )
$((wrap_if(:($i > $(loopentry-1)),
        quote
                        if ($ix>0 && $ix<=size($A,1) && $iy>0 && $iy<=size($A,2) && 0<$iz+$oz_max<=size($A,3)) 
                            $A_head[$tx,$ty] = $A[$ix,$iy,$iz+$oz_max] 
                        end
        end
        ;unless=(loopentry<=mainloopstart)
    )
    for (A, s) in shmem_vars if !use_shmemhalos[A] for (loopentry, oz_max,  tx, ty, nx_l, ny_l, A_head) = ((loopentrys[A], oz_maxs[A],  s[:tx], s[:ty], s[:nx_l], s[:ny_l], s[:A_head]),)
  )...
)
$(use_any_shmem ? 
    :(                  @sync_threads()
     ) :                NOEXPR
)
$((wrap_if(:($i > $(loopentry-1)),
       :(               $reg       = (0<$ix+$(oxy[1])<=size($A,1) && 0<$iy+$(oxy[2])<=size($A,2) && 0<$iz+$oz<=size($A,3)) ? $(regtarget(A, (oxy...,oz), indices)) : $reg
        )
        ;unless=(loopentry<=mainloopstart)
    )
    for A in optvars for (oxy, regs) in regqueue_heads[A] for (oz, reg) in regs for loopentry = (loopentrys[A],) if !use_shmems[A]
  )...
)
$((wrap_if(:($i > $(loopentry-1)),
    use_shmemhalo ? 
       :(               $reg       = $(regsource(A_head, oxy, (tx, ty)))                # e.g. A_ixm1_iyp2_izp3 = A_izp3[tx - 1, ty + 2]
        )
    :
       :(               $reg       = (0<$tx+$(oxy[1])<=$nx_l && 0<$ty+$(oxy[2])<=$ny_l) ? $(regsource(A_head, oxy, (tx, ty))) : (0<$ix+$(oxy[1])<=size($A,1) && 0<$iy+$(oxy[2])<=size($A,2) && 0<$iz+$oz<=size($A,3)) ? $(regtarget(A, (oxy...,oz), indices)) : $reg
        )
        ;unless=(loopentry<=mainloopstart)
    )
    for (A, s) in shmem_vars for (oxy, regs) in regqueue_heads[A] for (oz, reg) in regs for (use_shmemhalo, loopentry,  tx, ty, nx_l, ny_l, A_head) = ((use_shmemhalos[A], loopentrys[A],  s[:tx], s[:ty], s[:nx_l], s[:ny_l], s[:A_head]),)
  )...
)
$((wrap_if(:($i > 0),
        quote
                        $body
        end; 
        unless=(mainloopstart>=1)
    )
))
$(( # NOTE: the if statement is not needed here as we only deal with registers
    # wrap_if(:($i > $(loopentry-1)),
       :(
                        $(regs[oz]) = $(regs[oz+1])                                     # e.g. A_ixm1_iyp2_iz = A_ixm1_iyp2_izp1
        )
        # ;unless=(loopentry<=mainloopstart)
    # )
    for A in optvars for regs in values(regqueue_tails[A]) for oz in sort(keys(regs)) for (loopentry, oz_max) = ((loopentrys[A], oz_maxs[A]),) if oz<=oz_max-2
  )...
)
$((wrap_if(:($i > $(loopentry-1)),
    use_shmemhalo ? 
       :(               $reg       = $(regsource(A_head, oxy, (tx, ty)))                # e.g. A_ixm3_iyp2_izp2 = A_izp3[tx - 3, ty + 2]
        )
    :
       :(               $reg       = (0<$tx+$(oxy[1])<=$nx_l && 0<$ty+$(oxy[2])<=$ny_l) ? $(regsource(A_head, oxy, (tx, ty))) : (0<$ix+$(oxy[1])<=size($A,1) && 0<$iy+$(oxy[2])<=size($A,2) && 0<$iz+$oz<=size($A,3)) ? $(regtarget(A, (oxy...,oz), indices)) : $reg
        )
        ;unless=(loopentry<=mainloopstart)
    )
    for (A, s) in shmem_vars for (oxy, regs) in regqueue_tails[A] for (oz, reg) in regs for (use_shmemhalo, loopentry, oz_max,  tx, ty, nx_l, ny_l, A_head) = ((use_shmemhalos[A], loopentrys[A], oz_maxs[A],  s[:tx], s[:ty], s[:nx_l], s[:ny_l], s[:A_head]),) if oz==oz_max-1 && !(haskey(regqueue_heads[A], oxy) && haskey(regqueue_heads[A][oxy], oz_max))
  )...
)
# TODO: remove these as soon as the above is tested:
# $((wrap_if(:($i > $(loopentry-1)),
#        :(                $reg        = $(regsource(A_head, oxy, (tx, ty)))              # e.g. A_ixm3_iyp2_izp2 = A_izp3[tx - 3, ty + 2]
#         )
#         ;unless=(loopentry<=mainloopstart)
#     )
#     for (A, s) in shmem_vars for (oxy, regs) in regqueue_tails[A] for (oz, reg) in regs for (loopentry, oz_max, tx, ty, A_head) = ((loopentrys[A], oz_maxs[A], s[:tx], s[:ty], s[:A_head]),) if oz==oz_max-1 && !(haskey(regqueue_heads[A], oxy) && haskey(regqueue_heads[A][oxy], oz_max))
#   )...
# )
$(( # NOTE: the if statement is not needed here as we only deal with registers
    # wrap_if(:($i > $(loopentry-1)),
       :(
                        $reg           = $(regqueue_heads[A][oxy][oz_max])              # e.g. A_ixm1_iyp2_izp2 = A_ixm1_iyp2_izp3
        )
        # ;unless=(loopentry<=mainloopstart)
    # )
    for A in optvars for (oxy, regs) in regqueue_tails[A] for (oz, reg) in regs for (loopentry, oz_max) = ((loopentrys[A], oz_maxs[A]),) if oz==oz_max-1 && haskey(regqueue_heads[A], oxy) && haskey(regqueue_heads[A][oxy], oz_max)
  )...
)
        end
        # ;unroll=true
    ) # wrap_loop end
)                   # end

# Wrap-up-loop
#                     ParallelStencil.@unroll for $i = $(mainloopend+1):$loopend
#                         $tz_g = $i + $loopoffset
#                         if ($tz_g > $rangelength_z) ParallelStencil.@return_nothing; end
#                         $iz = ($tz_g < 1) ? $range_z_start-(1-$tz_g) : $range_z # TODO: this will probably always be formulated with range_z_start
# $((wrap_if(:($i > $(loopentry-1)),
#         quote
#                         @sync_threads()
#                         if (2*$t_h <= $nx_l*$ny_l && $ix_h>0 && $ix_h<=size($A,1) && $iy_h>0 && $iy_h<=size($A,2) && 0<$iz+$oz_max<=size($A,3)) 
#                             $A_head[$tx_h,$ty_h] = $A[$ix_h,$iy_h,$iz+$oz_max] 
#                         end
#                         if (2*$t_h2 <= $nx_l*$ny_l && $ix_h2>0 && $ix_h2<=size($A,1) && $iy_h2>0 && $iy_h2<=size($A,2) && 0<$iz+$oz_max<=size($A,3)) 
#                             $A_head[$tx_h2,$ty_h2] = $A[$ix_h2,$iy_h2,$iz+$oz_max]
#                         end
#                         @sync_threads()
#         end
#         ;unless=(loopentry<=mainloopstart)
#     )
#     for (A, s) in shmem_vars for (loopentry, oz_max,  tx, ty, nx_l, ny_l, t_h, t_h2, tx_h, tx_h2, ty_h, ty_h2, ix_h, ix_h2, iy_h, iy_h2, A_head) = ((loopentrys[A], oz_maxs[A],  s[:tx], s[:ty], s[:nx_l], s[:ny_l], s[:t_h], s[:t_h2], s[:tx_h], s[:tx_h2], s[:ty_h], s[:ty_h2], s[:ix_h], s[:ix_h2], s[:iy_h], s[:iy_h2], s[:A_head]),)
#   )...
# )
# $((wrap_if(:($i > $(loopentry-1)),
#        :(               $reg       = (0<$ix+$(oxy[1])<=size($A,1) && 0<$iy+$(oxy[2])<=size($A,2) && 0<$iz+$oz<=size($A,3)) ? $(regtarget(A, (oxy...,oz), indices)) : $reg
#         )
#         ;unless=(loopentry<=mainloopstart)
#     )
#     for A in optvars for (oxy, regs) in regqueue_heads[A] for (oz, reg) in regs for loopentry = (loopentrys[A],) if !use_shmems[A]
#   )...
# )
# $((wrap_if(:($i > $(loopentry-1)),
#        :(               $reg       = $(regsource(A_head, oxy, (tx, ty)))                # e.g. A_ixm1_iyp2_izp3 = A_izp3[tx - 1, ty + 2]
#         )
#         ;unless=(loopentry<=mainloopstart)
#     )
#     for (A, s) in shmem_vars for (oxy, regs) in regqueue_heads[A] for reg in values(regs) for (loopentry, tx, ty, A_head) = ((loopentrys[A], s[:tx], s[:ty], s[:A_head]),)
#   )...
# )
# $((wrap_if(:($i > 0),
#         quote
#                         $body
#         end; 
#         unless=(mainloopstart>=1)
#     )
# ))
# $((wrap_if(:($i > $(loopentry-1)),
#        :(
#                         $(regs[oz]) = $(regs[oz+1])                                     # e.g. A_ixm1_iyp2_iz = A_ixm1_iyp2_izp1
#         )
#         ;unless=(loopentry<=mainloopstart)
#     )
#     for A in optvars for regs in values(regqueue_tails[A]) for oz in sort(keys(regs)) for (loopentry, oz_max) = ((loopentrys[A], oz_maxs[A]),) if oz<=oz_max-2
#   )...
# )
# $((wrap_if(:($i > $(loopentry-1)),
#        :(                $reg        = $(regsource(A_head, oxy, (tx, ty)))              # e.g. A_ixm3_iyp2_izp2 = A_izp3[tx - 3, ty + 2]
#         )
#         ;unless=(loopentry<=mainloopstart)
#     )
#     for (A, s) in shmem_vars for (oxy, regs) in regqueue_tails[A] for (oz, reg) in regs for (loopentry, oz_max, tx, ty, A_head) = ((loopentrys[A], oz_maxs[A], s[:tx], s[:ty], s[:A_head]),) if oz==oz_max-1 && !(haskey(regqueue_heads[A], oxy) && haskey(regqueue_heads[A][oxy], oz_max))
#   )...
# )
# $((wrap_if(:($i > $(loopentry-1)),
#        :(
#                         $reg           = $(regqueue_heads[A][oxy][oz_max])              # e.g. A_ixm1_iyp2_izp2 = A_ixm1_iyp2_izp3
#         )
#         ;unless=(loopentry<=mainloopstart)
#     )
#     for A in optvars for (oxy, regs) in regqueue_tails[A] for (oz, reg) in regs for (loopentry, oz_max) = ((loopentrys[A], oz_maxs[A]),) if oz==oz_max-1 && haskey(regqueue_heads[A], oxy) && haskey(regqueue_heads[A][oxy], oz_max)
#   )...
# )

#                         $tz_g = $i + $loopoffset
#                         if ($tz_g > $rangelength_z) ParallelStencil.@return_nothing; end
#                         $iz = ($tz_g < 1) ? $range_z_start-(1-$tz_g) : $range_z # TODO: this will probably always be formulated with range_z_start
# $((
#     # wrap_if(:(($(loopentry-1) < $i < $(shmem_loopentry)) || ($(shmem_loopexit) < $i)),
#        :(               $reg       = (0<$ix+$(oxy[1])<=size($A,1) && 0<$iy+$(oxy[2])<=size($A,2) && 0<$iz+$oz<=size($A,3)) ? $(regtarget(A, (oxy...,oz), indices)) : $reg
#         )
#     for A in keys(shmem_vars) for (oxy, regs) in regqueue_heads[A] for (oz, reg) in regs for loopentry = (loopentrys[A],)
#   )...
# )
# $((
#        :(               $reg       = (0<$ix+$(oxy[1])<=size($A,1) && 0<$iy+$(oxy[2])<=size($A,2) && 0<$iz+$oz<=size($A,3)) ? $(regtarget(A, (oxy...,oz), indices)) : $reg
#         )
#     for A in optvars for (oxy, regs) in regqueue_heads[A] for (oz, reg) in regs for loopentry = (loopentrys[A],) if !use_shmems[A]
#   )...
# )
# $((
#         quote
#                         $body
#         end
# ))
# $((
#        :(
#                         $(regs[oz]) = $(regs[oz+1])                                     # e.g. A_ixm1_iyp2_iz = A_ixm1_iyp2_izp1
#         )
#     for A in optvars for regs in values(regqueue_tails[A]) for oz in sort(keys(regs)) for (loopentry, oz_max) = ((loopentrys[A], oz_maxs[A]),) if oz<=oz_max-2
#   )...
# )
# $((
#        :(
#                         $reg           = $(regqueue_heads[A][oxy][oz_max])              # e.g. A_ixm1_iyp2_izp2 = A_ixm1_iyp2_izp3
#         )
#     for A in optvars for (oxy, regs) in regqueue_tails[A] for (oz, reg) in regs for (loopentry, oz_max) = ((loopentrys[A], oz_maxs[A]),) if oz==oz_max-1 && haskey(regqueue_heads[A], oxy) && haskey(regqueue_heads[A][oxy], oz_max)
#   )...
# )
                    # end
        end
    else
        @ArgumentError("memopt: only optdim=3 is currently supported.")
    end
    store_metadata(metadata_module, is_parallel_kernel, offset_mins, offset_maxs, offsets, optvars, optdim, loopsize, optranges, use_shmemhalos)
    # @show QuoteNode(ParallelKernel.simplify_varnames!(ParallelKernel.remove_linenumbernodes!(deepcopy(body))))
    return body
end


function memopt(metadata_module::Module, is_parallel_kernel::Bool, caller::Module, indices::Union{Symbol,Expr}, optvars::Union{Expr,Symbol}, body::Expr; package::Symbol=get_package())
    optdim             = isa(indices,Expr) ? length(indices.args) : 1
    loopsize           = LOOPSIZE
    optranges          = nothing
    use_shmemhalos      = nothing
    optimize_halo_read = true
    return memopt(metadata_module, is_parallel_kernel, caller, indices, optvars, optdim, loopsize, optranges, use_shmemhalos, optimize_halo_read, body; package=package)
end


function shortif(else_val, if_expr; package::Symbol=get_package())
    if (package ∉ SUPPORTED_PACKAGES) @KeywordArgumentError("$ERRMSG_UNSUPPORTED_PACKAGE (obtained: $package).") end
    @capture(if_expr, if condition_ body_ end) || @ArgumentError("@shortif: the second argument must be an if statement.")
    @capture(body, lhs_ = rhs_) || @ArgumentError("@shortif: the if statement body must contain a assignement.")
    return :($lhs = $condition ? $rhs : $else_val)
end


function return_nothing()
    return :(return)
end


## FUNCTIONS FOR SHARED MEMORY ALLOCATION


## HELPER FUNCTIONS

function find_readonlyvars(body::Expr, indices::NTuple{N,<:Union{Symbol,Expr}} where N)
    vars         = Dict()
    writevars    = Dict()
    postwalk(body) do ex
        if is_stencil_access(ex, indices...)
            @capture(ex, A_[indices_expr__]) || @ModuleInternalError("a stencil access could not be pattern matched.")
            if haskey(vars, A) vars[A] += 1
            else               vars[A]  = 1
            end
        end
        if @capture(ex, (A_[indices_expr__] = rhs_) | (A_[indices_expr__] .= rhs_)) && is_stencil_access(:($A[$(indices_expr...)]), indices...)
            if haskey(writevars, A) writevars[A] += 1
            else                    writevars[A]  = 1
            end
        end
        return ex
    end
    readonlyvars = Dict(A => count for (A, count) in vars if A ∉ keys(writevars))
    return readonlyvars
end

function eval_offsets(caller::Module, body::Expr, indices::NTuple{N,<:Union{Symbol,Expr}} where N, int_type::Type{<:Integer})
    return postwalk(body) do ex
        if !is_stencil_access(ex, indices...) return ex; end
        @capture(ex, A_[indices_expr__]) || @ModuleInternalError("a stencil access could not be pattern matched.")
        for i = 1:length(indices)
            offset_expr = substitute(indices_expr[i], indices[i], 0)
            offset = eval_arg(caller, offset_expr)
            if     (offset >  0) indices_expr[i] = :($(indices[i]) + $(int_type(offset))     )
            elseif (offset <  0) indices_expr[i] = :($(indices[i]) - $(int_type(abs(offset))))
            else                 indices_expr[i] =     indices[i]
            end
        end
        return :($A[$(indices_expr...)])
    end
end

function extract_offsets(caller::Module, body::Expr, indices::NTuple{N,<:Union{Symbol,Expr}} where N, int_type::Type{<:Integer}, optvars::NTuple{N,Symbol} where N, optdim::Integer)
    offsets_by_xy = Dict(A => Dict() for A in optvars)
    offsets_by_z  = Dict(A => Dict() for A in optvars)
    postwalk(body) do ex
        if is_stencil_access(ex, indices...)
            @capture(ex, A_[indices_expr__]) || @ModuleInternalError("a stencil access could not be pattern matched.")
            if A in optvars
                offsets = ()
                for i = 1:length(indices)
                    offset_expr = substitute(indices_expr[i], indices[i], 0)
                    offset = int_type(eval_arg(caller, offset_expr)) # TODO: do this and cast later to enable unsigned integer (also dealing with negative rangers is required elsewhere): offset = eval_arg(caller, offset_expr)
                    offsets = (offsets..., offset)
                end
                if optdim == 3
                    k1 = offsets[1:2]
                    k2 = offsets[end]
                    if     haskey(offsets_by_xy[A], k1) && haskey(offsets_by_xy[A][k1], k2) offsets_by_xy[A][k1][k2] += 1
                    elseif haskey(offsets_by_xy[A], k1)                                     offsets_by_xy[A][k1][k2]  = 1
                    else                                                                    offsets_by_xy[A][k1]      = Dict(k2 => 1)
                    end
                    k1 = offsets[end]
                    k2 = offsets[1:2]
                    if     haskey(offsets_by_z[A], k1) && haskey(offsets_by_z[A][k1], k2) offsets_by_z[A][k1][k2] += 1
                    elseif haskey(offsets_by_z[A], k1)                                    offsets_by_z[A][k1][k2]  = 1
                    else                                                                  offsets_by_z[A][k1]      = Dict(k2 => 1)
                    end
                else
                    @ArgumentError("memopt: only optdim=3 is currently supported.")
                end
            end
        end
        return ex    
    end
    return offsets_by_xy, offsets_by_z
end

function remove_single_point_optvars(optvars, optranges_arg, offsets, offsets_by_z)
    return tuple((A for A in optvars if !(length(keys(offsets[A]))==1 && length(keys(offsets_by_z[A]))==1) || (!isnothing(optranges_arg) && A ∈ keys(optranges_arg)))...)
end

function define_optranges(optranges_arg, optvars, offsets, int_type)
    optranges = Dict()
    for A in optvars
        zspan_max     = 0
        oxy_zspan_max = ()
        for oxy in keys(offsets[A])
            zspan = length(keys(offsets[A][oxy]))
            if zspan > zspan_max
                zspan_max     = zspan
                oxy_zspan_max = oxy
            end
        end
        fullrange    = typemin(int_type):typemax(int_type)
        pointrange_x = oxy_zspan_max[1]: oxy_zspan_max[1]
        pointrange_y = oxy_zspan_max[2]: oxy_zspan_max[2]
        if     (!isnothing(optranges_arg) && A ∈ keys(optranges_arg)) optranges[A] = getproperty(optranges_arg, A)
        elseif (length(optvars) <= FULLRANGE_THRESHOLD)               optranges[A] = (fullrange,    fullrange,    fullrange)
        elseif (USE_FULLRANGE_DEFAULT == (true,  true,  true))        optranges[A] = (fullrange,    fullrange,    fullrange)
        elseif (USE_FULLRANGE_DEFAULT == (false, true,  true))        optranges[A] = (pointrange_x, fullrange,    fullrange)
        elseif (USE_FULLRANGE_DEFAULT == (true,  false, true))        optranges[A] = (fullrange,    pointrange_y, fullrange)
        elseif (USE_FULLRANGE_DEFAULT == (false, false, true))        optranges[A] = (pointrange_x, pointrange_y, fullrange)
        end
    end
    return optranges
end

function define_regqueues(offsets::Dict{Symbol, Dict{Any, Any}}, optranges::Dict{Any, Any}, optvars::NTuple{N,Symbol} where N, indices::NTuple{N,<:Union{Symbol,Expr}} where N, int_type::Type{<:Integer}, optdim::Integer)
    regqueue_heads = Dict(A => Dict() for A in optvars)
    regqueue_tails = Dict(A => Dict() for A in optvars)
    offset_mins    = Dict{Symbol, NTuple{3,Integer}}()
    offset_maxs    = Dict{Symbol, NTuple{3,Integer}}()
    nb_regs_heads  = Dict{Symbol, Integer}()
    nb_regs_tails  = Dict{Symbol, Integer}()
    for A in optvars
        regqueue_heads[A], regqueue_tails[A], offset_mins[A], offset_maxs[A], nb_regs_heads[A], nb_regs_tails[A] = define_regqueue(offsets[A], optranges[A], A, indices, int_type, optdim)
    end
    return regqueue_heads, regqueue_tails, offset_mins, offset_maxs, nb_regs_heads, nb_regs_tails
end

function define_regqueue(offsets::Dict{Any, Any}, optranges::NTuple{3,UnitRange}, A::Symbol, indices::NTuple{N,<:Union{Symbol,Expr}} where N, int_type::Type{<:Integer}, optdim::Integer)
    regqueue_head = Dict()
    regqueue_tail = Dict()
    nb_regs_head  = 0
    nb_regs_tail  = 0
    if optdim == 3
        optranges_xy     = optranges[1:2]
        optranges_z      = optranges[3]
        offsets_xy       = filter(oxy -> all(oxy .∈ optranges_xy), keys(offsets))
        if isempty(offsets_xy) @IncoherentArgumentError("incoherent argument in memopt: optranges in x-y dimension do not include any array access.") end
        offset_min       = (typemax(int_type), typemax(int_type), typemax(int_type))
        offset_max       = (typemin(int_type), typemin(int_type), typemin(int_type))
        for oxy in offsets_xy
            offsets_z = filter(x -> x ∈ optranges_z, keys(offsets[oxy]))
            if isempty(offsets_z) @IncoherentArgumentError("incoherent argument in memopt: optranges in z dimension do not include any array access.") end
            offset_min = (min(offset_min[1], oxy[1]),
                          min(offset_min[2], oxy[2]),
                          min(offset_min[3], minimum(offsets_z)))
            offset_max = (max(offset_max[1], oxy[1]),
                          max(offset_max[2], oxy[2]),
                          max(offset_max[3], maximum(offsets_z)))
        end
        oz_max = offset_max[3]
        for oxy in offsets_xy
            offsets_z = sort(filter(x -> x ∈ optranges_z, keys(offsets[oxy])))
            k1 = oxy
            for oz = offsets_z[1]:oz_max-1
                k2 = oz
                if haskey(regqueue_tail, k1) && haskey(regqueue_tail[k1], k2) @ModuleInternalError("regqueue_tail entry exists already.") end
                reg = gensym_world(varname(A, (oxy..., oz)), @__MODULE__);  nb_regs_tail += 1
                if haskey(regqueue_tail, k1) regqueue_tail[k1][k2] = reg
                else                         regqueue_tail[k1]     = Dict(k2 => reg)
                end
            end
            oz = offsets_z[end]
            if oz == oz_max
                k2 = oz
                if haskey(regqueue_head, k1) && haskey(regqueue_head[k1], k2) @ModuleInternalError("regqueue_head entry exists already.") end
                reg = gensym_world(varname(A, (oxy..., oz)), @__MODULE__);  nb_regs_head += 1
                if haskey(regqueue_head, k1) regqueue_head[k1][k2] = reg
                else                         regqueue_head[k1]     = Dict(k2 => reg)
                end
            end
        end
    else
        @ArgumentError("memopt: only optdim=3 is currently supported.")
    end
    return regqueue_head, regqueue_tail, offset_min, offset_max, nb_regs_head, nb_regs_tail
end

function define_helper_variables(offset_mins::Dict{Symbol, <:NTuple{3,Integer}}, offset_maxs::Dict{Symbol, <:NTuple{3,Integer}}, optvars::NTuple{N,Symbol} where N, use_shmemhalos_arg, optdim::Integer)
    oz_maxs, hx1s, hy1s, hx2s, hy2s, use_shmems, use_shmem_xs, use_shmem_ys, use_shmemhalos, use_shmemindices, offset_spans, oz_spans, loopentrys = Dict(), Dict(), Dict(), Dict(), Dict(), Dict(), Dict(), Dict(), Dict(), Dict(), Dict(), Dict(), Dict()
    if optdim == 3
        for A in optvars
            offset_min, offset_max = offset_mins[A], offset_maxs[A]
            oz_max         = offset_max[3]
            hx1, hy1       = -1 .* offset_min[1:2]
            hx2, hy2       = offset_max[1:2]
            use_shmem_x    = (hx1 + hx2 > 0)
            use_shmem_y    = (hy1 + hy2 > 0)
            use_shmem      = use_shmem_x || use_shmem_y
            use_shmemhalo  = if (!isnothing(use_shmemhalos_arg) && (A ∈ keys(use_shmemhalos_arg))) getproperty(use_shmemhalos_arg, A)
                             elseif !(use_shmem_x && use_shmem_y)                                  USE_SHMEMHALO_1D_DEFAULT
                             else                                                                  USE_SHMEMHALO_DEFAULT
                             end
            use_shmemindex = use_shmem && use_shmemhalo && (use_shmem_x && use_shmem_y)
            offset_span    = offset_max .- offset_min
            oz_span        = offset_span[3]
            loopentry      = 1 - oz_span #TODO: make possibility to do first and last read in z dimension directly into registers without halo
            oz_maxs[A], hx1s[A], hy1s[A], hx2s[A], hy2s[A], use_shmems[A], use_shmem_xs[A], use_shmem_ys[A], use_shmemhalos[A], use_shmemindices[A], offset_spans[A], oz_spans[A], loopentrys[A] = oz_max, hx1, hy1, hx2, hy2, use_shmem, use_shmem_x, use_shmem_y, use_shmemhalo, use_shmemindex, offset_span, oz_span, loopentry
        end
    else
        @ArgumentError("memopt: only optdim=3 is currently supported.")
    end
    return oz_maxs, hx1s, hy1s, hx2s, hy2s, use_shmems, use_shmem_xs, use_shmem_ys, use_shmemhalos, use_shmemindices, offset_spans, oz_spans, loopentrys
end

function define_shmem_index_groups(hx1s, hy1s, hx2s, hy2s, optvars::NTuple{N,Symbol} where N, use_shmems::Dict{Any, Any}, optdim::Integer)
    shmem_index_groups = Dict()
    if optdim == 3
        for A in optvars
            if use_shmems[A]
                k = (hx1s[A], hy1s[A], hx2s[A], hy2s[A])
                if !haskey(shmem_index_groups, k) shmem_index_groups[k] = (A,)
                else                              shmem_index_groups[k] = (shmem_index_groups[k]..., A)
                end
            end
        end
    end
    return shmem_index_groups
end

function define_shmem_vars(oz_maxs::Dict{Any, Any}, hx1s, hy1s, hx2s, hy2s, optvars::NTuple{N,Symbol} where N, indices, use_shmems::Dict{Any, Any}, use_shmem_xs, use_shmem_ys, shmem_index_groups, use_shmemhalos, use_shmemindices, optdim::Integer)
    ix, iy, iz = indices
    shmem_vars = Dict(A => Dict() for A in optvars if use_shmems[A])
    if optdim == 3
        for vars in values(shmem_index_groups)
            suffix = join(string.(vars), "_")
            sym_tx    = gensym_world("tx_$suffix", @__MODULE__)
            sym_ty    = gensym_world("ty_$suffix", @__MODULE__)
            sym_nx_l  = gensym_world("nx_l_$suffix", @__MODULE__)
            sym_ny_l  = gensym_world("ny_l_$suffix", @__MODULE__)
            sym_t_h   = gensym_world("t_h_$suffix", @__MODULE__)
            sym_t_h2  = gensym_world("t_h2_$suffix", @__MODULE__)
            sym_tx_h  = gensym_world("tx_h_$suffix", @__MODULE__)
            sym_tx_h2 = gensym_world("tx_h2_$suffix", @__MODULE__)
            sym_ty_h  = gensym_world("ty_h_$suffix", @__MODULE__)
            sym_ty_h2 = gensym_world("ty_h2_$suffix", @__MODULE__)
            sym_ix_h  = gensym_world("ix_h_$suffix", @__MODULE__)
            sym_ix_h2 = gensym_world("ix_h2_$suffix", @__MODULE__)
            sym_iy_h  = gensym_world("iy_h_$suffix", @__MODULE__)
            sym_iy_h2 = gensym_world("iy_h2_$suffix", @__MODULE__)
            for A in vars   
                if use_shmemindices[A]
                    n_l = quote $sym_nx_l*$sym_ny_l end
                    shmem_vars[A][:tx]     = sym_tx
                    shmem_vars[A][:ty]     = sym_ty
                    shmem_vars[A][:nx_l]   = sym_nx_l
                    shmem_vars[A][:ny_l]   = sym_ny_l
                    shmem_vars[A][:n_l]    = n_l
                    shmem_vars[A][:t_h]    = sym_t_h
                    shmem_vars[A][:t_h2]   = sym_t_h2
                    shmem_vars[A][:tx_h]   = sym_tx_h
                    shmem_vars[A][:tx_h2]  = sym_tx_h2
                    shmem_vars[A][:ty_h]   = sym_ty_h
                    shmem_vars[A][:ty_h2]  = sym_ty_h2
                    shmem_vars[A][:ix_h]   = sym_ix_h
                    shmem_vars[A][:ix_h2]  = sym_ix_h2
                    shmem_vars[A][:iy_h]   = sym_iy_h
                    shmem_vars[A][:iy_h2]  = sym_iy_h2
                else
                    if use_shmemhalos[A]
                        use_shmem_x, use_shmem_y = use_shmem_xs[A], use_shmem_ys[A]
                        hx1, hy1, hx2, hy2 = hx1s[A], hy1s[A], hx2s[A], hy2s[A]
                        if use_shmem_x && use_shmem_y # NOTE: if the following expressions are noted with ":()" then it will cause a segmentation fault and run time.
                            tx    = quote @threadIdx().x + $hx1 end
                            ty    = quote @threadIdx().y + $hy1 end
                            nx_l  = quote @blockDim().x + $(hx1+hx2) end
                            ny_l  = quote @blockDim().y + $(hy1+hy2) end
                            n_l   = quote $nx_l*$ny_l end
                            t_h   = quote (@threadIdx().y-1)*@blockDim().x + @threadIdx().x end  # NOTE: here it must be bx, not @blockDim().x
                            t_h2  = quote $t_h + $nx_l*$ny_l - @blockDim().x*@blockDim().y end
                            ty_h  = quote ($t_h-1) ÷ $nx_l + 1 end
                            tx_h  = quote ($t_h-1) % $nx_l + 1 end                               # NOTE: equivalent to (worse performance has uses registers probably differently): ($t_h-1) - $nx_l*($ty_h-1) + 1
                            ty_h2 = quote ($t_h2-1) ÷ $nx_l + 1 end
                            tx_h2 = quote ($t_h2-1) % $nx_l + 1 end                              # NOTE: equivalent to (worse performance has uses registers probably differently): ($t_h2-1) - $nx_l*($ty_h2-1) + 1
                            ix_h  = quote $ix - @threadIdx().x + $tx_h  - $hx1 end    # NOTE: here it must be @blockDim().x, not bx
                            ix_h2 = quote $ix - @threadIdx().x + $tx_h2 - $hx1 end    # ...
                            iy_h  = quote $iy - @threadIdx().y + $ty_h  - $hy1 end    # ...
                            iy_h2 = quote $iy - @threadIdx().y + $ty_h2 - $hy1 end    # ...
                        elseif use_shmem_x
                            tx    = quote @threadIdx().x + $hx1 end
                            ty    = quote @threadIdx().y + $hy1 end
                            nx_l  = quote @blockDim().x + $(hx1+hx2) end
                            ny_l  = quote @blockDim().y end
                            tx_h  = quote @threadIdx().x end
                            ty_h  = quote @threadIdx().y end
                            tx_h2 = quote @threadIdx().x + $(hx1+hx2) end # NOTE: alternative: shmem_vars[A][:tx_h2]  = :(@threadIdx().x + @blockDim().x)
                            ty_h2 = ty_h
                            ix_h  = quote $ix - @threadIdx().x + $tx_h  - $hx1 end
                            ix_h2 = quote $ix - @threadIdx().x + $tx_h2 - $hx1 end
                            iy_h  = quote $iy - @threadIdx().y + $ty_h  - $hy1 end
                            iy_h2 = quote $iy - @threadIdx().y + $ty_h2 - $hy1 end
                            n_l   = nx_l
                            t_h   = tx_h
                            t_h2  = tx_h2
                        elseif use_shmem_y
                            tx    = quote @threadIdx().x + $hx1 end
                            ty    = quote @threadIdx().y + $hy1 end
                            nx_l  = quote @blockDim().x end
                            ny_l  = quote @blockDim().y + $(hy1+hy2) end
                            tx_h  = quote @threadIdx().x end
                            ty_h  = quote @threadIdx().y end
                            tx_h2 = tx_h
                            ty_h2 = quote @threadIdx().y + $(hy1+hy2) end # NOTE: alternative: # shmem_vars[A][:ty_h2]  = :(@threadIdx().y + @blockDim().y)
                            ix_h  = quote $ix - @threadIdx().x + $tx_h  - $hx1 end
                            ix_h2 = quote $ix - @threadIdx().x + $tx_h2 - $hx1 end
                            iy_h  = quote $iy - @threadIdx().y + $ty_h  - $hy1 end
                            iy_h2 = quote $iy - @threadIdx().y + $ty_h2 - $hy1 end
                            n_l   = ny_l
                            t_h   = ty_h
                            t_h2  = ty_h2
                        end
                        shmem_vars[A][:tx]    = tx
                        shmem_vars[A][:ty]    = ty
                        shmem_vars[A][:nx_l]  = nx_l
                        shmem_vars[A][:ny_l]  = ny_l
                        shmem_vars[A][:n_l]   = n_l
                        shmem_vars[A][:t_h]   = t_h
                        shmem_vars[A][:t_h2]  = t_h2
                        shmem_vars[A][:tx_h]  = tx_h
                        shmem_vars[A][:tx_h2] = tx_h2
                        shmem_vars[A][:ty_h]  = ty_h
                        shmem_vars[A][:ty_h2] = ty_h2
                        shmem_vars[A][:ix_h]  = ix_h
                        shmem_vars[A][:ix_h2] = ix_h2
                        shmem_vars[A][:iy_h]  = iy_h
                        shmem_vars[A][:iy_h2] = iy_h2
                    else
                        shmem_vars[A][:tx]     = :(@threadIdx().x)
                        shmem_vars[A][:ty]     = :(@threadIdx().y)
                        shmem_vars[A][:nx_l]   = :(@blockDim().x)
                        shmem_vars[A][:ny_l]   = :(@blockDim().y)
                    end
                end
                shmem_vars[A][:A_head] = gensym_world(varname(A, (oz_maxs[A],); i="iz"), @__MODULE__)
            end
        end
    else
        @ArgumentError("memopt: only optdim=3 is currently supported.")
    end
    return shmem_vars
end

function define_shmem_exprs(shmem_vars::Dict{Symbol, Dict{Any, Any}}, optdim::Integer)
    exprs = Dict(A => Dict() for A in keys(shmem_vars))
    offset = ()
    if optdim == 3
        for A in keys(shmem_vars)
            exprs[A][:offset] = (length(offset) > 0) ? Expr(:call, :+, offset...) : 0
            offset = (offset..., :($(shmem_vars[A][:nx_l]) * $(shmem_vars[A][:ny_l]) * sizeof(eltype($A))))
        end
    else
        @ArgumentError("memopt: only optdim=3 is currently supported.")
    end
    return exprs
end

function define_shmem_z_ranges(offsets_by_z::Dict{Symbol, Dict{Any, Any}}, use_shmems::Dict{Any, Any}, optdim::Integer)
    shmem_z_ranges = Dict()
    shmem_As = (A for (A, use_shmem) in use_shmems if use_shmem)
    for A in shmem_As
        shmem_z_ranges[A] = define_shmem_z_range(offsets_by_z[A], optdim)
    end
    return shmem_z_ranges
end

function define_shmem_z_range(offsets_by_z::Dict{Any, Any}, optdim::Integer)
    start, start_offsets_xy = find_rangelimit(offsets_by_z, optdim; upper=false)
    stop,  stop_offsets_xy  = find_rangelimit(offsets_by_z, optdim; upper=true)
    if (length(start_offsets_xy) != 1 || length(stop_offsets_xy) != 1 || start_offsets_xy[1] != stop_offsets_xy[1]) # NOTE: shared memory range is not reduced in asymmetric case
        return minimum(keys(offsets_by_z)):maximum(keys(offsets_by_z))
    end
    return start:stop
end

function find_rangelimit(offsets_by_z::Dict{Any, Any}, optdim::Integer; upper=false)
    if optdim == 3
        offsets_z   = sort(keys(offsets_by_z); rev=upper)
        oz1         = offsets_z[1]
        rangelimit  = oz1
        offsets_xy1 = (keys(offsets_by_z[oz1])...,)
        if length(offsets_xy1) == 1
            rangelimit = offsets_z[2]
            oxy1 = offsets_xy1[1]
            for oz in offsets_z[2:end]
                offsets_xy = (keys(offsets_by_z[oz])...,)
                if (length(offsets_xy) == 1) && (offsets_xy[1] == oxy1)
                    rangelimit = offsets_z[oz+1]
                else
                    break
                end
            end
        end
    else
        @ArgumentError("memopt: only optdim=3 is currently supported.")
    end
    return rangelimit, offsets_xy1
end

function define_shmem_loopentrys(loopentrys, shmem_z_ranges, offset_mins, optdim::Integer)
    shmem_loopentrys = Dict()
    shmem_As = (A for A in keys(shmem_z_ranges))
    for A in shmem_As
        shmem_loopentrys[A] = define_shmem_loopentry(loopentrys[A], shmem_z_ranges[A], offset_mins[A], optdim)
    end
    return shmem_loopentrys
end

function define_shmem_loopentry(loopentry, shmem_z_range, offset_min, optdim::Integer)
    if optdim == 3
        shmem_loopentry = loopentry + (shmem_z_range.start - offset_min[3])
    else
        @ArgumentError("memopt: only optdim=3 is currently supported.")
    end
    return shmem_loopentry
end

function define_shmem_loopexits(loopexit, shmem_z_ranges, offset_maxs, optdim::Integer)
    shmem_loopexits = Dict()
    shmem_As = (A for A in keys(shmem_z_ranges))
    for A in shmem_As
        shmem_loopexits[A] = define_shmem_loopexit(loopexit, shmem_z_ranges[A], offset_maxs[A], optdim)
    end
    return shmem_loopexits
end

function define_shmem_loopexit(loopexit, shmem_z_range, offset_max, optdim::Integer)
    if optdim == 3
        shmem_loopexit = loopexit - (offset_max[3] - shmem_z_range.stop)
    else
        @ArgumentError("memopt: only optdim=3 is currently supported.")
    end
    return shmem_loopexit
end

function varname(A::Symbol, offsets::NTuple{N,Integer} where N; i::String="ix", j::String="iy", k::String="iz")
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

function regtarget(A::Symbol, offsets::NTuple{N,Integer} where N, indices::NTuple{N,<:Union{Symbol,Expr}} where N)
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

function regsource(A_head::Symbol, offsets::NTuple{N,Integer} where N, local_indices::NTuple{N,<:Union{Symbol,Expr}} where N)
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

function wrap_if(condition::Expr, block::Expr; unless::Bool=false)
    if unless
        return block
    else
        return quote 
                    if $condition
                        $block
                    end
                end
    end
end

function wrap_loop(index::Symbol, range::UnitRange, block::Expr; unroll=false)
    if unroll
        return quote
                    $(( quote
                            $index = $i
                            $block
                        end
                        for i in range
                    )...
                    )
                end
    else
        return quote
                    for $index = $(range.start):$(range.stop)
                        $block
                    end
                end
    end
end

function store_metadata(metadata_module::Module, is_parallel_kernel::Bool, offset_mins::Dict{Symbol, <:NTuple{3,Integer}}, offset_maxs::Dict{Symbol, <:NTuple{3,Integer}}, offsets::Dict{Symbol, Dict{Any, Any}}, optvars::NTuple{N,Symbol} where N, optdim::Integer, loopsize::Integer, optranges::Dict{Any, Any}, use_shmemhalos)
    storeexpr = quote
        const is_parallel_kernel = $is_parallel_kernel
        const memopt            = true
        const stencilranges      = $(NamedTuple(A => (offset_mins[A][1]:offset_maxs[A][1], offset_mins[A][2]:offset_maxs[A][2], offset_mins[A][3]:offset_maxs[A][3]) for A in optvars))
        const offsets            = $offsets
        const optvars            = $optvars
        const optdim             = $optdim
        const loopsize           = $loopsize
        const optranges          = $optranges
        const use_shmemhalos     = $use_shmemhalos
    end
    @eval(metadata_module, $storeexpr)
end

Base.sort(keys::T; kwargs...) where T<:Base.AbstractSet = sort([keys...]; kwargs...)


# macro unroll(args...) check_initialized(); checkargs_unroll(args...); esc(unroll(args...)); end

# function checkargs_unroll(args...)
#     if (length(args) != 1) @ArgumentError("wrong number of arguments.") end
# end

# function unroll(expr)
#     if @capture(expr, for i_ = range_ body__ end) #TODO: enable in instead of equal
#         return quote
#             for $i = $range
#                 $(body...)
#                 $(Expr(:loopinfo, nodes...))
#             end
#         end
#     else
#         error("Syntax error: loopinfo needs a for loop")
#     end
# end
