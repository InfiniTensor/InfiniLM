//! Common types and functions used in transformer.

#![deny(warnings, missing_docs)]

/// `utok` for token id.
#[allow(non_camel_case_types)]
pub type utok = u32;

/// `upos` for position id.
#[allow(non_camel_case_types)]
pub type upos = u32;

mod gguf;
pub mod test_model;

pub use gguf::{map_files, GGufModel, GGufTensor};

use std::{
    alloc::{alloc, dealloc, Layout},
    mem::align_of,
    ops::{Deref, DerefMut},
    ptr::NonNull,
    slice::{from_raw_parts, from_raw_parts_mut},
};

/// A wrapper around a dynamically allocated byte array.
pub struct Blob {
    ptr: NonNull<u8>,
    len: usize,
}

unsafe impl Send for Blob {}
unsafe impl Sync for Blob {}

impl Blob {
    /// Creates a new `Blob` with the given size.
    ///
    /// The allocated block of memory may or may not be initialized.
    #[inline]
    pub fn new(size: usize) -> Self {
        const ALIGN: usize = align_of::<usize>();
        let layout = Layout::from_size_align(size, ALIGN).unwrap();
        Self {
            ptr: NonNull::new(unsafe { alloc(layout) }).unwrap(),
            len: size,
        }
    }
}

impl Drop for Blob {
    #[inline]
    fn drop(&mut self) {
        const ALIGN: usize = align_of::<usize>();
        let layout = Layout::from_size_align(self.len, ALIGN).unwrap();
        unsafe { dealloc(self.ptr.as_ptr(), layout) }
    }
}

impl Deref for Blob {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for Blob {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}
