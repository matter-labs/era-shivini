use era_cudart::result::CudaResult;
use era_cudart_sys::CudaError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DryRunState {
    Stopped,
    Running,
    Failing(CudaError),
}

use DryRunState::*;

static mut DRY_RUN_STATE: DryRunState = Stopped;

pub(crate) fn dry_run_start() {
    unsafe {
        assert_eq!(DRY_RUN_STATE, Stopped);
        DRY_RUN_STATE = Running;
    }
}

pub(crate) fn dry_run_stop() -> CudaResult<()> {
    unsafe {
        assert_ne!(DRY_RUN_STATE, Stopped);
        let state = DRY_RUN_STATE;
        DRY_RUN_STATE = Stopped;
        if let Failing(e) = state {
            Err(e)
        } else {
            Ok(())
        }
    }
}

pub(crate) fn dry_run_fail(error: CudaError) {
    unsafe {
        assert_ne!(DRY_RUN_STATE, Stopped);
        DRY_RUN_STATE = Failing(error);
    }
}

pub(crate) fn is_dry_run() -> CudaResult<bool> {
    unsafe {
        match DRY_RUN_STATE {
            Stopped => Ok(false),
            Running => Ok(true),
            Failing(e) => Err(e),
        }
    }
}

macro_rules! if_not_dry_run {
    ($($t:tt)*) => {
        if !crate::primitives::dry_run::is_dry_run()? {
            $($t)*
        }
        else {
            Ok(())
        }
    };
}

pub(crate) use if_not_dry_run;
