use mpi::ffi;
use mpi::topology::SimpleCommunicator;
use mpi::traits::*;
use std::collections::HashMap;
use std::mem;
use std::thread;
use libc::c_void;

use rest_tensors::RIFull;

use crate::mpi_io::MPIOperator;

const TAG_REQUEST: i32 = 100;
const TAG_BLOCK:   i32 = 101;
pub struct ServicePortal<'a> {
    pub mpi_op: &'a MPIOperator,
    pub data: &'a HashMap<usize, RIFull<f64>>,
    pending_fetch_requests: Vec<ffi::MPI_Request>,   // Irecv + Isend for own fetches
    pending_data_sends: Vec<ffi::MPI_Request>,       // Isends we issued while serving
    done_request_started: bool,
    done_request: ffi::MPI_Request,
    local_done: i32,
    global_done: i32,
}

impl<'a> ServicePortal<'a> {
    pub fn new(mpi_op: &'a MPIOperator, data: &'a HashMap<usize, RIFull<f64>>) -> Self {
        ServicePortal
        {
            mpi_op: mpi_op,
            data: data,
            pending_fetch_requests: Vec::new(),
            pending_data_sends: Vec::new(),
            done_request_started: false,
            done_request: unsafe { mem::zeroed() },
            local_done: 0,
            global_done: 0,
        }
    }

    pub fn request_block(&mut self, block_id: usize, owner: i32, buffer: &mut [f64]) {
        let mut recv_req: ffi::MPI_Request = unsafe { mem::zeroed() };
        let mut send_req: ffi::MPI_Request = unsafe { mem::zeroed() };
        let blk = block_id as i32;
        unsafe {
            ffi::MPI_Irecv(
                buffer.as_mut_ptr() as *mut c_void,
                buffer.len() as i32,
                ffi::RSMPI_DOUBLE,
                owner,
                TAG_BLOCK,
                self.mpi_op.world.as_raw(),
                &mut recv_req,
            );

            ffi::MPI_Send(
                &blk as *const i32 as *const c_void,
                1,
                ffi::RSMPI_INT32_T,
                owner,
                TAG_REQUEST,
                self.mpi_op.world.as_raw(),
            );
        }
        self.pending_fetch_requests.push(recv_req);

    }

    pub fn wait_for_fetches_with_service(&mut self) {
        while !self.pending_fetch_requests.is_empty() {
            // Handle incoming block requests from other ranks
            self.service();

            let mut flag: i32 = 0;
            unsafe {
                ffi::MPI_Testall(
                    self.pending_fetch_requests.len() as i32,
                    self.pending_fetch_requests.as_mut_ptr(),
                    &mut flag,
                    ffi::RSMPI_STATUSES_IGNORE,
                );
            }

            if flag != 0 {
                // Fetches are complete. Drain any requests that arrived just
                // before the test to ensure the queue is empty.
                self.service();
                self.pending_fetch_requests.clear();
                break;
            }

            // Optional: yield to avoid busy-waiting
            std::thread::yield_now();
        }
    }

    pub fn service(&mut self) {
        let mut flag: i32 = 0;
        let mut status: ffi::MPI_Status = unsafe { std::mem::zeroed() };

        loop {
            unsafe {
                ffi::MPI_Iprobe(
                    ffi::RSMPI_ANY_SOURCE,
                    TAG_REQUEST,
                    self.mpi_op.world.as_raw(),
                    &mut flag,
                    &mut status,
                );
            }
            if flag == 0 {
                break;
            }

            let mut block_id: i32 = 0;
            unsafe {
                ffi::MPI_Recv(
                    &mut block_id as *mut i32 as *mut c_void,
                    1,
                    ffi::RSMPI_INT32_T,
                    status.MPI_SOURCE,
                    TAG_REQUEST,
                    self.mpi_op.world.as_raw(),
                    ffi::RSMPI_STATUS_IGNORE,
                );
            }

            let blk = block_id as usize;
            let tensor = match self.data.get(&blk) {
                Some(t) => t,
                None => {
                    eprintln!(
                        "Rank {}: block {} requested but not owned locally.",
                        self.mpi_op.rank, blk
                    );
                    unsafe { ffi::MPI_Abort(self.mpi_op.world.as_raw(), 1) };
                    unreachable!();
                }
            };

            let data_ptr = tensor.data.as_ptr() as *const f64;
            let block_size = tensor.data.len() as i32;

            let mut sreq: ffi::MPI_Request = unsafe { std::mem::zeroed() };
            unsafe {
                ffi::MPI_Isend(
                    data_ptr as *const c_void,
                    block_size,
                    ffi::RSMPI_DOUBLE,
                    status.MPI_SOURCE,
                    TAG_BLOCK,
                    self.mpi_op.world.as_raw(),
                    &mut sreq,
                );
            }
            self.pending_data_sends.push(sreq);
        }

        // Progress non‑blocking allreduce if started
        if self.done_request_started {
            let mut done_flag: i32 = 0;
            unsafe {
                ffi::MPI_Test(
                    &mut self.done_request as *mut ffi::MPI_Request,
                    &mut done_flag,
                    ffi::RSMPI_STATUS_IGNORE,
                );
            }
        }
    }


    pub fn signal_done(&mut self) {
        if !self.done_request_started {
            self.local_done = 1;
            unsafe {
                ffi::MPI_Iallreduce(
                    &self.local_done as *const i32 as *const c_void,
                    &mut self.global_done as *mut i32 as *mut c_void,
                    1,
                    ffi::RSMPI_INT32_T,
                    ffi::RSMPI_SUM,
                    self.mpi_op.world.as_raw(),
                    &mut self.done_request,
                );
            }
            self.done_request_started = true;
        }
    }

    pub fn is_done(&mut self) -> bool {
        if !self.done_request_started {
            return false;
        }
        let mut flag: i32 = 0;
        unsafe {
            ffi::MPI_Test(
                &mut self.done_request as *mut ffi::MPI_Request,
                &mut flag,
                ffi::RSMPI_STATUS_IGNORE,
            );
        }
        flag != 0
    }

    pub fn finalize(&mut self) {
        if !self.pending_data_sends.is_empty() {
            unsafe {
                ffi::MPI_Waitall(
                    self.pending_data_sends.len() as i32,
                    self.pending_data_sends.as_mut_ptr(),
                    ffi::RSMPI_STATUSES_IGNORE,
                );
            }
            self.pending_data_sends.clear();
        }
    }
}
