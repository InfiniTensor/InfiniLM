#![cfg(detected_neuware)]
// Include the bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

mod gather;
mod sample;

use cndrv::{ContextSpore, CurrentCtx, DevByte};
use common::utok;
pub use operators::{cndrv, cambricon_mlu::Handle as Mlu};
pub use sample::sample_cpu;

// pub type CTensor = Tensor;
// use tensor::Tensor;
use digit_layout::DigitLayout;
use operators::{cndrv::AsRaw, QueueOf};
use std::ops::{Deref, DerefMut};

pub use tensor::{Tensor as rustTensor, slice};
pub use common_devices::{Kernels, KernelsA, KernelsB, SliceOn};

impl DataLayout {
    pub fn new(packed: u16, sign: u16, size: u16, mantissa: u16, exponent: u16) -> Self {
        DataLayout {
            _bitfield_align_1: [0; 0],
            _bitfield_1: Self::new_bitfield_1(packed, sign, size, mantissa, exponent),
        }
    }
}

impl From<DigitLayout> for DataLayout {
    fn from(digit_layout: DigitLayout) -> Self {
        // 根据实际的 DigitLayout 类型进行转换
        DataLayout::new(
            digit_layout.packed() as u16,
            digit_layout.signed() as u16,
            digit_layout.nbytes() as u16,
            digit_layout.mantissa() as u16,
            digit_layout.exponent() as u16,
        )
    }
}

fn to_ctensor<T>(tensor: &rustTensor<T>) -> Tensor
where
    T: Deref<Target = [DevByte]>,
{
    // 获取 strides
    let strides_vec: Vec<i64> = tensor.strides().iter().map(|&x| x as i64).collect();
    let strides_ptr: *mut i64 = strides_vec.as_ptr() as *mut i64;

    // 获取 shape
    let shape_vec: Vec<u64> = tensor.shape().iter().map(|&x| x as u64).collect();
    let shape_ptr: *mut u64 = shape_vec.as_ptr() as *mut u64;

    unsafe {
        // 创建 TensorDescriptor
        let mut descriptor = std::ptr::null_mut();
        let datatype = DataLayout::from(tensor.data_layout());
        createTensorDescriptor(&mut descriptor, tensor.shape().len() as u64, shape_ptr, strides_ptr, datatype);
    
        if !descriptor.is_null() {
            // 获取数据指针
            let data = tensor.physical().as_ptr() as *mut std::ffi::c_void;
            Tensor {
                layout: descriptor,
                data,
            }
        } else {
            panic!("Failed to create TensorDescriptor");
        }

    }
}

pub struct CambriconKernels {
    mat_mul: *mut MatmulDescriptor,
    rms_norm: *mut RMSNormDescriptor,
    rope: *mut RotaryEmbeddingDescriptor,
    reform: *mut ReformDescriptor,
    softmax: *mut CausalSoftmaxDescriptor,
    swiglu: *mut SwigluDescriptor,    
}


impl CambriconKernels {   
    pub fn new(device: DeviceEnum) -> Self {
        let config: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            Self {
                mat_mul: createMatmulDescriptor(device, config) as *mut MatmulDescriptor,
                rms_norm: createRMSNormDescriptor(device, config) as *mut RMSNormDescriptor,
                rope: createRotaryEmbeddingDescriptor(device, config) as *mut RotaryEmbeddingDescriptor,
                reform: createReformDescriptor(device, config) as *mut ReformDescriptor,
                softmax: createCausalSoftmaxDescriptor(device, config) as *mut CausalSoftmaxDescriptor,
                swiglu: createSwigluDescriptor(device, config) as *mut SwigluDescriptor,
            }
        }
    }
}

impl Kernels<Mlu> for CambriconKernels {}

impl KernelsA for CambriconKernels {
    type Handle = Mlu;

    fn rms_norm<T, U, V>(
        &self,
        y: &mut rustTensor<T>,
        x: &rustTensor<U>,
        w: &rustTensor<V>,
        epsilon: f32,
        queue: &QueueOf<Self::Handle>,
    ) where
    T: DerefMut<Target = SliceOn<Self::Handle>>,
    U: Deref<Target = SliceOn<Self::Handle>>,
    V: Deref<Target = SliceOn<Self::Handle>>
    {
        unsafe {    
            let y = to_ctensor(y);
            let x = to_ctensor(x);
            let w = to_ctensor(w);
    
            rmsNorm(self.rms_norm, y, x, w, epsilon, queue.as_raw() as *mut ::std::os::raw::c_void);
    
            // Destroy the SwigluDescriptor
            destroyTensorDescriptor(x.layout);
            destroyTensorDescriptor(y.layout);
            destroyTensorDescriptor(w.layout);
        }
    }

    fn rope<T, U>(
        &self,
        t: &mut rustTensor<T>,
        pos: &rustTensor<U>,
        theta: f32,
        queue: &QueueOf<Self::Handle>,
    ) where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
        U: Deref<Target = SliceOn<Self::Handle>>,
    {
        unsafe {
            let t = to_ctensor(t);
            let pos = to_ctensor(pos);
    
            rotaryEmbedding(self.rope, t, pos, theta, queue.as_raw() as *mut ::std::os::raw::c_void);
    
            // Destroy the SwigluDescriptor
            destroyTensorDescriptor(t.layout);
            destroyTensorDescriptor(pos.layout);
        }
    }

    fn mat_mul<T, U, V>(
        &self,
        c: &mut rustTensor<T>,
        beta: f32,
        a: &rustTensor<U>,
        b: &rustTensor<V>,
        alpha: f32,
        queue: &QueueOf<Self::Handle>,
    ) where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
        U: Deref<Target = SliceOn<Self::Handle>>,
        V: Deref<Target = SliceOn<Self::Handle>>,
    {
        unsafe {
            let c = to_ctensor(c);
            let a = to_ctensor(a);
            let b = to_ctensor(b);
    
            matmul(self.mat_mul, c, beta, a, b, alpha, queue.as_raw() as *mut ::std::os::raw::c_void);
    
            // Destroy the SwigluDescriptor
            destroyTensorDescriptor(c.layout);
            destroyTensorDescriptor(a.layout);
            destroyTensorDescriptor(b.layout);
        }
    }

    fn softmax<T>(&self, att: &mut rustTensor<T>, queue: &QueueOf<Self::Handle>)
    where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
    {
        unsafe {
            let att = to_ctensor(att);
    
            causalSoftmax(self.softmax, att, queue.as_raw() as *mut ::std::os::raw::c_void);
    
            // Destroy the SwigluDescriptor
            destroyTensorDescriptor(att.layout);
        }
    }

    fn swiglu<T, U>(&self, gate: &mut rustTensor<T>, up: &rustTensor<U>, queue: &QueueOf<Self::Handle>)
    where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
        U: Deref<Target = SliceOn<Self::Handle>>,
    {
        unsafe {
            let gate = to_ctensor(gate);
            let up = to_ctensor(up);
    
            swiglu(self.swiglu, gate, up, queue.as_raw() as *mut ::std::os::raw::c_void);
    
            // Destroy the SwigluDescriptor
            destroyTensorDescriptor(gate.layout);
            destroyTensorDescriptor(up.layout);
        }
    }    

}

impl KernelsB for CambriconKernels {
    type Handle = Mlu;

    fn gather<T, U, I>(
        &self,
        x: &mut rustTensor<T>,
        table: &rustTensor<U>,
        tokens: I,
        queue: &QueueOf<Self::Handle>,
    ) where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
        U: Deref<Target = [u8]>,
        I: IntoIterator<Item = utok>,
    {
        gather::gather(x, table, tokens, queue);
    }


    fn reform<T, U>(&self, dst: &mut rustTensor<T>, src: &rustTensor<U>, queue: &QueueOf<Self::Handle>)
    where
        T: DerefMut<Target = SliceOn<Self::Handle>>,
        U: Deref<Target = SliceOn<Self::Handle>>,
    {
        unsafe { 
            let dst = to_ctensor(dst);
            let src = to_ctensor(src);
    
            reform(self.reform, dst, src, queue.as_raw() as *mut ::std::os::raw::c_void);
    
            // Destroy the SwigluDescriptor
            destroyTensorDescriptor(dst.layout);
            destroyTensorDescriptor(src.layout);
        }
    }


}

pub struct DropOption<T>(Option<T>);

impl<T> From<T> for DropOption<T> {
    #[inline]
    fn from(value: T) -> Self {
        Self(Some(value))
    }
}
 
impl<T> AsRef<T> for DropOption<T> {
    #[inline]
    fn as_ref(&self) -> &T {
        self.0.as_ref().unwrap()
    }
}

impl<T> AsMut<T> for DropOption<T> {
    #[inline]
    fn as_mut(&mut self) -> &mut T {
        self.0.as_mut().unwrap()
    }
}

impl<T: ContextSpore> DropOption<T> {
    #[inline]
    pub fn sprout<'ctx>(&mut self, ctx: &'ctx CurrentCtx) -> <T as ContextSpore>::Resource<'ctx> {
        self.0.take().unwrap().sprout(ctx)
    }
}

pub fn synchronize() {
    cndrv::init();
    for i in 0..cndrv::Device::count() {
        cndrv::Device::new(i as _)
            .acquire_shared()
            .apply(|ctx| ctx.synchronize());
    }
}
