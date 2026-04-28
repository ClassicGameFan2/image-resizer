// =============================================================================
// fsr3_pass_new_locks.cpp
// FSR 3.1.5 CPU Port — Pass 9: New Locks
//
// Source: ffx_fsr3upscaler_new_locks.h / ffx_fsr3upscaler_lock_pass.hlsl
//
// FSR3.1 CHANGE:
//   The lock pass is split into two sub-passes:
//     Pass 2 (fsr3PassLock): Computes luma-stability-based lock values at
//                            display res from the current jittered frame.
//     Pass 9 (fsr3PassNewLocks): Computes "new lock" candidates — pixels that
//                            are becoming stable for the first time (or
//                            returning to stability after disocclusion).
//
//   New locks are pixels where:
//     - The current lock value is high (stable pixel)
//     - The previous accumulated lock was low (not previously locked)
//   This prevents already-locked stable pixels from being "re-locked" every
//   frame, which would cause the lock luminance to drift.
//
//   For static images with jitter OFF: all stable pixels become new locks
//   on frame 0, and lock for the rest of the frames.
//   With jitter ON: new locks emerge as the jitter pattern stabilizes.
// =============================================================================
#include "fsr3_types.h"
#include "fsr_math.h"
#include <cmath>
#include <algorithm>

void fsr3PassNewLocks(
    const float*         colorBuffer,  // render res, float RGBA (unused here,
    Fsr3InternalBuffers& buf)          //   lock was already computed in pass 2)
{
    int dW = buf.displayW, dH = buf.displayH;

    for (int y = 0; y < dH; y++) {
        for (int x = 0; x < dW; x++) {
            size_t idx     = (size_t)y * dW + x;
            float curLock  = buf.lockMask[idx];
            // Previous history lock (from prevAccumulatedColor alpha channel
            // is not used here — we use the separate lockAccum concept).
            // In our CPU port: newLockMask = max(0, curLock - prevHistory).
            // Simplified: a pixel is a "new lock" if it's currently stable
            // and wasn't fully stable in the accumulated buffer yet.
            // We approximate by looking at the accumulationFactor:
            // low accumulationFactor → still converging → potentially new lock.
            float prevAccum = buf.prevAccumulationFactor[idx];

            // New lock = stable now AND wasn't fully accumulated before
            // The 0.5 threshold matches the FSR3.1 shader heuristic.
            float newLock = (curLock > 0.5f && prevAccum < 0.8f)
                ? curLock : 0.0f;

            buf.newLockMask[idx] = newLock;
        }
    }
}
