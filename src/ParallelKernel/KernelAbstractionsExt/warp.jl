## WARP-LEVEL PRIMITIVES (mapped to the sub-group functions of KernelInterface)
#
# See the section "KERNELABSTRACTIONS BACKEND: WARP-LEVEL PRIMITIVES" in ParallelKernel/kernel_language.jl for
# the semantics. In short: a warp is a KernelInterface sub-group, lanes are 1-based, `mask` is ignored (all lanes
# of the sub-group have to participate), and the shuffles always use KernelInterface's `width` variants, which
# give the CUDA semantics (also for the full sub-group width when no `width` is given).

import ParallelStencil.ParallelKernel: warpsize_kernelabstractions, laneid_kernelabstractions, active_mask_kernelabstractions,
                                       shfl_sync_kernelabstractions, shfl_up_sync_kernelabstractions, shfl_down_sync_kernelabstractions, shfl_xor_sync_kernelabstractions,
                                       vote_any_sync_kernelabstractions, vote_all_sync_kernelabstractions, vote_ballot_sync_kernelabstractions

@static if isdefined(KernelAbstractions, :KernelInterface)

const KI = KernelAbstractions.KernelInterface

@inline warpsize_kernelabstractions() = KI.get_max_sub_group_size()

@inline laneid_kernelabstractions() = KI.get_sub_group_local_id()

# The lanes that exist in the sub-group, i.e. bits 0:get_sub_group_size()-1 (shifting a UInt64 by 64 gives 0, so a full
# sub-group of 64 lanes gives typemax(UInt64)). KernelInterface has no notion of the active lanes of a divergent branch.
@inline active_mask_kernelabstractions() = (UInt64(1) << (KI.get_sub_group_size() % UInt64)) - UInt64(1)

@inline shfl_sync_kernelabstractions(mask, val, lane::Integer)                      = KI.shfl(val, lane, KI.get_max_sub_group_size())
@inline shfl_sync_kernelabstractions(mask, val, lane::Integer, width::Integer)      = KI.shfl(val, lane, width)
@inline shfl_up_sync_kernelabstractions(mask, val, delta::Integer)                  = KI.shfl_up(val, delta, KI.get_max_sub_group_size())
@inline shfl_up_sync_kernelabstractions(mask, val, delta::Integer, width::Integer)  = KI.shfl_up(val, delta, width)
@inline shfl_down_sync_kernelabstractions(mask, val, delta::Integer)                = KI.shfl_down(val, delta, KI.get_max_sub_group_size())
@inline shfl_down_sync_kernelabstractions(mask, val, delta::Integer, width::Integer)= KI.shfl_down(val, delta, width)
@inline shfl_xor_sync_kernelabstractions(mask, val, lane_mask::Integer)                = KI.shfl_xor(val, lane_mask, KI.get_max_sub_group_size())
@inline shfl_xor_sync_kernelabstractions(mask, val, lane_mask::Integer, width::Integer)= KI.shfl_xor(val, lane_mask, width)

@inline vote_any_sync_kernelabstractions(mask, predicate::Bool)    = KI.sub_group_any(predicate)
@inline vote_all_sync_kernelabstractions(mask, predicate::Bool)    = KI.sub_group_all(predicate)
@inline vote_ballot_sync_kernelabstractions(mask, predicate::Bool) = KI.sub_group_ballot(predicate)

else # KernelAbstractions < 0.10: no sub-group support.

using ParallelStencil.ParallelKernel: ERRMSG_KERNELABSTRACTIONS_WARP
warpsize_kernelabstractions()                          = @ArgumentError(ERRMSG_KERNELABSTRACTIONS_WARP)
laneid_kernelabstractions()                            = @ArgumentError(ERRMSG_KERNELABSTRACTIONS_WARP)
active_mask_kernelabstractions()                       = @ArgumentError(ERRMSG_KERNELABSTRACTIONS_WARP)
shfl_sync_kernelabstractions(mask, val, x, w...)        = @ArgumentError(ERRMSG_KERNELABSTRACTIONS_WARP)
shfl_up_sync_kernelabstractions(mask, val, x, w...)     = @ArgumentError(ERRMSG_KERNELABSTRACTIONS_WARP)
shfl_down_sync_kernelabstractions(mask, val, x, w...)   = @ArgumentError(ERRMSG_KERNELABSTRACTIONS_WARP)
shfl_xor_sync_kernelabstractions(mask, val, x, w...)    = @ArgumentError(ERRMSG_KERNELABSTRACTIONS_WARP)
vote_any_sync_kernelabstractions(mask, predicate)      = @ArgumentError(ERRMSG_KERNELABSTRACTIONS_WARP)
vote_all_sync_kernelabstractions(mask, predicate)      = @ArgumentError(ERRMSG_KERNELABSTRACTIONS_WARP)
vote_ballot_sync_kernelabstractions(mask, predicate)   = @ArgumentError(ERRMSG_KERNELABSTRACTIONS_WARP)

end
