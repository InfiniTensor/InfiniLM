use crate::{idim, pattern::Pattern, udim, Shape};
use digit_layout::DigitLayout;
use nalgebra::DVector;
use operators::{Operator, TensorLayout};
use std::{
    mem::{align_of, size_of},
    ops::{Deref, DerefMut},
};

#[derive(Clone, Debug)]
pub struct Tensor<Physical> {
    pub(crate) layout: DigitLayout,
    pub(crate) shape: Shape,
    pub(crate) pattern: Pattern,
    pub(crate) physical: Physical,
}

impl<Physical> Tensor<Physical> {
    #[inline]
    pub fn new(layout: DigitLayout, shape: &[udim], physical: Physical) -> Self {
        Self {
            layout,
            pattern: Pattern::from_shape(shape, 0),
            shape: Shape::from_slice(shape),
            physical,
        }
    }

    #[inline]
    pub fn alloc(
        data_type: DigitLayout,
        shape: &[udim],
        f: impl FnOnce(usize) -> Physical,
    ) -> Self {
        Self {
            layout: data_type,
            pattern: Pattern::from_shape(shape, 0),
            shape: Shape::from_slice(shape),
            physical: f(shape.iter().product::<udim>() as usize * data_type.nbytes()),
        }
    }

    /// # Safety
    ///
    /// The caller must ensure that the parts are valid.
    #[inline]
    pub unsafe fn from_raw_parts(
        data_type: DigitLayout,
        shape: &[udim],
        pattern: &[idim],
        physical: Physical,
    ) -> Self {
        Self {
            layout: data_type,
            shape: shape.iter().copied().collect(),
            pattern: Pattern(DVector::from_vec(pattern.to_vec())),
            physical,
        }
    }

    #[inline]
    pub const fn data_layout(&self) -> DigitLayout {
        self.layout
    }

    #[inline]
    pub fn shape(&self) -> &[udim] {
        &self.shape
    }

    #[inline]
    pub fn pattern(&self) -> &[idim] {
        self.pattern.0.as_slice()
    }

    #[inline]
    pub fn strides(&self) -> &[idim] {
        self.pattern.strides()
    }

    #[inline]
    pub fn bytes_offset(&self) -> isize {
        self.pattern.offset() as isize * self.layout.nbytes() as isize
    }

    #[inline]
    pub const fn physical(&self) -> &Physical {
        &self.physical
    }

    #[inline]
    pub fn physical_mut(&mut self) -> &mut Physical {
        &mut self.physical
    }

    #[inline]
    pub fn size(&self) -> usize {
        self.shape.iter().map(|&d| d as usize).product()
    }

    #[inline]
    pub fn bytes_size(&self) -> usize {
        self.size() * self.layout.nbytes()
    }

    #[inline]
    pub fn is_contiguous(&self) -> bool {
        self.contiguous_len() == self.shape.len()
    }

    /// 连续维度的数量。
    pub fn contiguous_len(&self) -> usize {
        self.pattern
            .strides()
            .iter()
            .enumerate()
            .rev()
            .scan(1 as idim, |mul, (i, &s)| {
                if s == *mul || s == 0 {
                    *mul *= self.shape[i] as idim;
                    Some(())
                } else {
                    None
                }
            })
            .count()
    }

    #[inline]
    pub fn as_ref(&self) -> Tensor<&Physical> {
        Tensor {
            layout: self.layout,
            shape: self.shape.clone(),
            pattern: self.pattern.clone(),
            physical: &self.physical,
        }
    }

    #[inline]
    pub fn as_mut(&mut self) -> Tensor<&mut Physical> {
        Tensor {
            layout: self.layout,
            shape: self.shape.clone(),
            pattern: self.pattern.clone(),
            physical: &mut self.physical,
        }
    }

    #[inline]
    pub fn take_physical(self) -> Physical {
        self.physical
    }

    #[inline]
    pub fn map_physical<U>(self, f: impl FnOnce(Physical) -> U) -> Tensor<U> {
        Tensor {
            layout: self.layout,
            shape: self.shape,
            pattern: self.pattern,
            physical: f(self.physical),
        }
    }

    pub fn layout(&self) -> TensorLayout {
        let dt = self.data_layout();
        let shape = &self.shape;
        let strides = self.strides();

        let d = dt.nbytes() as i64;

        let mut ans = TensorLayout::new(dt, shape.len());
        let (shape_, strides_) = ans.as_mut();
        for i in 0..self.shape.len() {
            shape_[i] = (shape[i] as u64).into();
            strides_[i] = (strides[i] as i64 * d).into();
        }
        ans
    }
}

impl<B: Sized, P: Deref<Target = [B]>> Tensor<P> {
    pub fn base(&self) -> *const B {
        const { assert!(size_of::<B>() == 1) }
        const { assert!(align_of::<B>() == 1) }

        let off = self.bytes_offset();
        unsafe { self.physical.as_ptr().cast::<u8>().offset(off).cast() }
    }
}

impl<B: Sized, P: DerefMut<Target = [B]>> Tensor<P> {
    pub fn base_mut(&mut self) -> *mut B {
        const { assert!(size_of::<B>() == 1) }
        const { assert!(align_of::<B>() == 1) }

        let off = self.bytes_offset();
        unsafe {
            self.physical_mut()
                .as_mut_ptr()
                .cast::<u8>()
                .offset(off)
                .cast()
        }
    }
}

impl<Physical: Deref<Target = [u8]>> Tensor<Physical> {
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        debug_assert!(self.is_contiguous());
        let off = self.bytes_offset();
        let len = self.bytes_size();
        &self.physical[off as usize..][..len]
    }

    /// # Safety
    ///
    /// The caller must ensure that the `dst` can be a valid tensor physical.
    pub unsafe fn reform_to_raw(&self, dst: &mut [u8]) {
        assert_eq!(self.bytes_size(), dst.len());
        use operators::{
            common_cpu::{Handle as Cpu, ThisThread},
            reform::{common_cpu::Operator as Reform, Args},
        };
        Reform::new(&Cpu)
            .launch(
                &Args {
                    dst_layout: {
                        let dt = self.data_layout();
                        let shape = &self.shape;

                        let mut mul = dt.nbytes() as i64;

                        let mut ans = TensorLayout::new(dt, shape.len());
                        let (shape_, strides_) = ans.as_mut();
                        for i in (0..self.shape.len()).rev() {
                            shape_[i] = (shape[i] as u64).into();
                            strides_[i] = mul.into();
                            mul *= shape[i] as i64;
                        }
                        ans
                    },
                    dst_base: dst.as_mut_ptr(),
                    src_layout: self.layout(),
                    src_base: self.base(),
                },
                &ThisThread,
            )
            .unwrap();
    }

    pub fn reform_to<U>(&self, dst: &mut Tensor<U>)
    where
        U: DerefMut<Target = [u8]>,
    {
        use operators::{
            common_cpu::{Handle as Cpu, ThisThread},
            reform::{common_cpu::Operator as Reform, Args},
        };
        Reform::new(&Cpu)
            .launch(
                &Args {
                    dst_layout: dst.layout(),
                    dst_base: dst.base_mut(),
                    src_layout: self.layout(),
                    src_base: self.base(),
                },
                &ThisThread,
            )
            .unwrap();
    }
}

impl<Physical: DerefMut<Target = [u8]>> Tensor<Physical> {
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        debug_assert!(self.is_contiguous());
        let off = self.bytes_offset();
        let len = self.bytes_size();
        &mut self.physical[off as usize..][..len]
    }
}

#[test]
fn test() {
    use digit_layout::types::F32;

    let t = Tensor::new(F32, &[2, 3, 4, 5], ());
    assert_eq!(t.shape(), &[2, 3, 4, 5]);
    assert_eq!(t.pattern.0.as_slice(), &[60, 20, 5, 1, 0]);
    assert_eq!(t.contiguous_len(), 4);
    assert_eq!(t.is_contiguous(), true);

    let t = t.reshape(&[2, 3, 20]);
    assert_eq!(t.shape(), &[2, 3, 20]);
    assert_eq!(t.pattern.0.as_slice(), &[60, 20, 1, 0]);
    assert_eq!(t.contiguous_len(), 3);
    assert_eq!(t.is_contiguous(), true);

    let t = t.transpose(&[1, 0, 2]);
    assert_eq!(t.shape(), &[3, 2, 20]);
    assert_eq!(t.pattern.0.as_slice(), &[20, 60, 1, 0]);
    assert_eq!(t.contiguous_len(), 1);
    assert_eq!(t.is_contiguous(), false);

    let t = t.reshape(&[3, 1, 1, 2, 5, 1, 4, 1, 1, 1]);
    assert_eq!(t.shape(), &[3, 1, 1, 2, 5, 1, 4, 1, 1, 1]);
    assert_eq!(t.pattern.0.as_slice(), &[20, 0, 0, 60, 4, 0, 1, 0, 0, 0, 0]);
    assert_eq!(t.contiguous_len(), 6);
    assert_eq!(t.is_contiguous(), false);

    let t = t.reshape(&[3, 2, 1, 5, 2, 2]);
    assert_eq!(t.shape(), &[3, 2, 1, 5, 2, 2]);
    assert_eq!(t.pattern.0.as_slice(), &[20, 60, 0, 4, 2, 1, 0]);
    assert_eq!(t.contiguous_len(), 4);
    assert_eq!(t.is_contiguous(), false);
}
