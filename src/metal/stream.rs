//! A resident arena of expert slots filled from the SSD while the model runs.
//!
//! The routed experts of a model too large for memory stay in the file. Each
//! slot holds one expert's matrices (`parts` of `part_bytes` each); part `p`
//! of every slot lives in its own region, so region `p` is laid out as an
//! ordinary stacked expert matrix whose expert index is the slot. The
//! expert kernels read it unchanged, given slot ids in place of expert ids.
//!
//! Host writes into a buffer the GPU also reads are the one thing the rest of
//! the backend never does, so the arena enforces its own rule: a slot is
//! written only when no work that may read it is in flight. A slot is pinned
//! before work that reads it is encoded, and stays busy until the device's
//! next completion wait (every wait drains all submitted work). Reads into
//! the arena run on I/O threads that get only addresses; the arena waits for
//! them before it can be dropped, and its bytes are never mapped for reading.
use super::{MetalBuffer, MetalDevice};
use crate::{Error, Result};
use objc2_metal::MTLBuffer;
use std::{
    cell::RefCell,
    fs::File,
    os::{fd::AsRawFd, unix::fs::FileExt},
    rc::Rc,
    sync::{Arc, Condvar, Mutex, mpsc},
    thread::JoinHandle,
};

unsafe extern "C" {
    fn fcntl(fd: i32, cmd: i32, ...) -> i32;
}
/// `fcntl` command that turns the unified buffer cache off for a descriptor.
const F_NOCACHE: i32 = 48;

/// Read `len` bytes at `offset` of file `file` into part `part` of `slot`.
#[derive(Clone, Copy, Debug)]
pub struct SlotRead {
    pub slot: usize,
    pub part: usize,
    pub file: usize,
    pub offset: u64,
}

struct Job {
    file: Arc<File>,
    offset: u64,
    address: usize,
    len: usize,
    batch: Arc<Batch>,
}

#[derive(Default)]
struct Batch {
    state: Mutex<(usize, Option<String>)>,
    done: Condvar,
}

impl Batch {
    fn finish(&self, error: Option<String>) {
        let mut state = self.state.lock().expect("batch lock");
        state.0 -= 1;
        if state.1.is_none() {
            state.1 = error;
        }
        if state.0 == 0 {
            self.done.notify_all();
        }
    }
    fn wait(&self) -> Option<String> {
        let mut state = self.state.lock().expect("batch lock");
        while state.0 > 0 {
            state = self.done.wait(state).expect("batch lock");
        }
        state.1.take()
    }
}

/// Reads submitted together; `ExpertArena::wait` completes them.
pub struct Ticket {
    batch: Arc<Batch>,
    slots: Vec<usize>,
}

pub struct ExpertArena {
    buffer: Rc<MetalBuffer>,
    part_bytes: usize,
    parts: usize,
    slots: usize,
    files: Vec<Arc<File>>,
    /// Completion wait count when each slot was last pinned.
    pinned: RefCell<Vec<Option<usize>>>,
    /// Slots with reads submitted and not yet waited for.
    filling: RefCell<Vec<bool>>,
    /// Tickets not yet waited for; drained before the buffer can go.
    open: RefCell<Vec<Arc<Batch>>>,
    jobs: Option<mpsc::Sender<Job>>,
    threads: Vec<JoinHandle<()>>,
}

impl ExpertArena {
    /// An arena of `slots` slots of `parts` matrices of `part_bytes` each,
    /// read from `files` (opened without the buffer cache) by `threads` I/O
    /// threads.
    pub fn new(
        device: &MetalDevice,
        part_bytes: usize,
        parts: usize,
        slots: usize,
        files: Vec<File>,
        threads: usize,
    ) -> Result<Self> {
        let bytes = part_bytes
            .checked_mul(parts)
            .and_then(|b| b.checked_mul(slots))
            .filter(|&b| b > 0)
            .ok_or_else(|| Error::Parameter("empty or oversized expert arena".into()))?;
        let buffer = Rc::new(device.allocate_resident(bytes)?);
        for file in &files {
            // SAFETY: F_NOCACHE takes an integer argument and only changes the
            // caching policy of a descriptor this process owns.
            let r = unsafe { fcntl(file.as_raw_fd(), F_NOCACHE, 1i32) };
            if r == -1 {
                return Err(Error::Io(std::io::Error::last_os_error()));
            }
        }
        let (tx, rx) = mpsc::channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        let threads = (0..threads.max(1))
            .map(|_| {
                let rx = rx.clone();
                std::thread::spawn(move || {
                    loop {
                        let job = match rx.lock().expect("job queue").recv() {
                            Ok(job) => job,
                            Err(_) => return,
                        };
                        // SAFETY: `address..address + len` lies inside the
                        // arena's allocation (checked in `submit`), which the
                        // arena keeps alive until every open batch is waited
                        // for. The slot was idle on the GPU and not being
                        // filled by another job when the read was submitted,
                        // and the arena never forms a reference to its bytes,
                        // so this is the only reference to them.
                        let dst = unsafe {
                            std::slice::from_raw_parts_mut(job.address as *mut u8, job.len)
                        };
                        let error = job
                            .file
                            .read_exact_at(dst, job.offset)
                            .err()
                            .map(|e| format!("expert read at {}: {e}", job.offset));
                        job.batch.finish(error);
                    }
                })
            })
            .collect();
        Ok(Self {
            buffer,
            part_bytes,
            parts,
            slots,
            files: files.into_iter().map(Arc::new).collect(),
            pinned: RefCell::new(vec![None; slots]),
            filling: RefCell::new(vec![false; slots]),
            open: RefCell::new(Vec::new()),
            jobs: Some(tx),
            threads,
        })
    }

    pub fn slots(&self) -> usize {
        self.slots
    }
    pub fn part_bytes(&self) -> usize {
        self.part_bytes
    }
    pub fn byte_size(&self) -> usize {
        self.buffer.len_bytes()
    }

    /// Region `part`: buffer, byte offset and length, as a stacked matrix of
    /// `slots` experts.
    pub(crate) fn region(&self, part: usize) -> (Rc<MetalBuffer>, usize, usize) {
        assert!(part < self.parts);
        let len = self.part_bytes * self.slots;
        (self.buffer.clone(), part * len, len)
    }

    /// Whether `slot` may be read by work still in flight on `device`.
    pub fn busy(&self, device: &MetalDevice, slot: usize) -> bool {
        self.pinned.borrow()[slot] == Some(device.counters().completion_waits)
    }

    /// Mark `slots` as read by work about to be encoded. Their reads must
    /// have been waited for.
    pub fn pin(&self, device: &MetalDevice, slots: &[usize]) -> Result<()> {
        let epoch = device.counters().completion_waits;
        let filling = self.filling.borrow();
        let mut pinned = self.pinned.borrow_mut();
        for &s in slots {
            if s >= self.slots || filling[s] {
                return Err(Error::Dispatch(format!(
                    "expert slot {s} is out of range or still being read"
                )));
            }
            pinned[s] = Some(epoch);
        }
        Ok(())
    }

    /// Start `reads` on the I/O threads. Each target slot must be idle on the
    /// GPU and not already being filled.
    pub fn submit(&self, device: &MetalDevice, reads: &[SlotRead]) -> Result<Ticket> {
        let epoch = device.counters().completion_waits;
        let batch = Arc::new(Batch::default());
        batch.state.lock().expect("batch lock").0 = reads.len();
        let mut slots = Vec::with_capacity(reads.len());
        {
            let pinned = self.pinned.borrow();
            let mut filling = self.filling.borrow_mut();
            for r in reads {
                if r.slot >= self.slots || r.part >= self.parts || r.file >= self.files.len() {
                    return Err(Error::Range("expert read outside the arena".into()));
                }
                if pinned[r.slot] == Some(epoch) {
                    return Err(Error::Dispatch(format!(
                        "expert slot {} is in use by work in flight",
                        r.slot
                    )));
                }
                if filling[r.slot] && !slots.contains(&r.slot) {
                    return Err(Error::Dispatch(format!(
                        "expert slot {} is already being read",
                        r.slot
                    )));
                }
                filling[r.slot] = true;
                if !slots.contains(&r.slot) {
                    slots.push(r.slot);
                }
            }
        }
        // Several parts of one slot are distinct ranges; one part twice is not.
        let mut targets: Vec<(usize, usize)> = reads.iter().map(|r| (r.slot, r.part)).collect();
        targets.sort_unstable();
        if targets.windows(2).any(|w| w[0] == w[1]) {
            let mut filling = self.filling.borrow_mut();
            for &s in &slots {
                filling[s] = false;
            }
            return Err(Error::Dispatch("one expert slot part read twice".into()));
        }
        let base = self.buffer.raw.contents().as_ptr() as usize;
        let jobs = self.jobs.as_ref().expect("arena is open");
        for r in reads {
            let offset = r.part * self.part_bytes * self.slots + r.slot * self.part_bytes;
            jobs.send(Job {
                file: self.files[r.file].clone(),
                offset: r.offset,
                address: base + offset,
                len: self.part_bytes,
                batch: batch.clone(),
            })
            .map_err(|_| Error::Dispatch("expert I/O threads stopped".into()))?;
        }
        self.open.borrow_mut().push(batch.clone());
        Ok(Ticket { batch, slots })
    }

    /// Complete a ticket's reads. On error the slots' contents are unknown;
    /// the caller must not map them as the experts they were meant to hold.
    pub fn wait(&self, ticket: Ticket) -> Result<()> {
        let error = ticket.batch.wait();
        self.open
            .borrow_mut()
            .retain(|b| !Arc::ptr_eq(b, &ticket.batch));
        let mut filling = self.filling.borrow_mut();
        for s in ticket.slots {
            filling[s] = false;
        }
        match error {
            Some(e) => Err(Error::Io(std::io::Error::other(e))),
            None => Ok(()),
        }
    }
}

impl Drop for ExpertArena {
    fn drop(&mut self) {
        // Reads in flight write into the buffer: finish them before it goes.
        for batch in self.open.borrow_mut().drain(..) {
            batch.wait();
        }
        self.jobs.take();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}
