use operators::cuda::{Device, MemProp, PhyMem, VirMem};
use std::{
    ops::{Range, RangeBounds},
    sync::Arc,
};

pub(crate) struct MemPages {
    prop: MemProp,
    size: usize,
    pool: Vec<Arc<PhyMem>>,
}

impl MemPages {
    pub fn new(dev: Device) -> Self {
        let prop = dev.mem_prop();
        let size = prop.granularity_minimum();
        let pool = Vec::new();
        Self { prop, size, pool }
    }

    #[inline(always)]
    pub const fn page_size(&self) -> usize {
        self.size
    }

    #[inline]
    pub fn reserve_vir(&self, len: usize) -> VirMem {
        VirMem::new(len.div_ceil(self.size) * self.size, 0)
    }

    pub fn map(&mut self, mem: &mut VirMem, range: impl RangeBounds<usize>) {
        for i in self.page_range(mem, range) {
            mem.map(i * self.size, self.take());
        }
    }

    pub fn unmap(&mut self, mem: &mut VirMem, range: impl RangeBounds<usize>) {
        for i in self.page_range(mem, range) {
            self.pool.push(mem.unmap(i * self.size))
        }
    }

    #[inline]
    fn take(&mut self) -> Arc<PhyMem> {
        self.pool
            .pop()
            .unwrap_or_else(|| self.prop.create(self.size))
    }

    fn page_range(&self, mem: &VirMem, range: impl RangeBounds<usize>) -> Range<usize> {
        use std::ops::Bound::{Excluded, Included, Unbounded};
        let start = match range.start_bound() {
            Included(i) => *i,
            Excluded(i) => *i + 1,
            Unbounded => 0,
        };
        let end = match range.end_bound() {
            Included(i) => *i + 1,
            Excluded(i) => *i,
            Unbounded => mem.len() / self.size,
        };
        start..end
    }
}
