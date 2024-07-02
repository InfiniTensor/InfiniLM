#![cfg(detected_neuware)]
// Include the bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
pub extern crate cndrv;

use cndrv::{ContextGuard, ContextSpore};
use common::utok;

// pub type CTensor = Tensor;
// use tensor::Tensor;
use digit_layout::DigitLayout;
use std::ops::{Deref, DerefMut};

pub use tensor::Tensor as rustTensor;
pub use common_devices::{Kernels, SliceOn};

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
    T: Deref<Target = [u8]>,
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

pub struct CambriconKernels;

impl CambriconKernels {

    fn gather<T, U, I>(
        &self,
        x: &mut rustTensor<T>,
        table: &rustTensor<U>,
        tokens: I,
        stream: *mut ::std::os::raw::c_void,
    ) where
        T: DerefMut<Target = [u8]>,
        U: Deref<Target = [u8]>,
        I: IntoIterator<Item = utok>,
    {
        todo!()
    }

    fn rms_norm<T, U, V>(
        &self,
        y: &mut rustTensor<T>,
        x: &rustTensor<U>,
        w: &rustTensor<V>,
        epsilon: f32,
        stream: *mut ::std::os::raw::c_void,
    ) where
        T: DerefMut<Target = [u8]>,
        U: Deref<Target = [u8]>,
        V: Deref<Target = [u8]>,
    {
        let device = DeviceEnum::DevCpu;
        let config: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let descriptor = createRMSNormDescriptor(device, config) as *mut RMSNormDescriptor;
    
            let y = to_ctensor(y);
            let x = to_ctensor(x);
            let w = to_ctensor(w);
    
            rmsNorm(descriptor, y, x, w, epsilon, stream);
    
            // Destroy the SwigluDescriptor
            destroyRMSNormDescriptor(descriptor);
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
        stream: *mut ::std::os::raw::c_void,
    ) where
        T: DerefMut<Target = [u8]>,
        U: Deref<Target = [u8]>,
    {
        let device = DeviceEnum::DevCpu;
        let config: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let descriptor = createRotaryEmbeddingDescriptor(device, config) as *mut RotaryEmbeddingDescriptor;
    
            let t = to_ctensor(t);
            let pos = to_ctensor(pos);
    
            rotaryEmbedding(descriptor, t, pos, theta, stream);
    
            // Destroy the SwigluDescriptor
            destroyRotaryEmbeddingDescriptor(descriptor);
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
        stream: *mut ::std::os::raw::c_void,
    ) where
        T: DerefMut<Target = [u8]>,
        U: Deref<Target = [u8]>,
        V: Deref<Target = [u8]>,
    {
        let device = DeviceEnum::DevCpu;
        let config: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let descriptor = createMatmulDescriptor(device, config) as *mut MatmulDescriptor;
    
            let c = to_ctensor(c);
            let a = to_ctensor(a);
            let b = to_ctensor(b);
    
            matmul(descriptor, c, beta, a, b, alpha, stream);
    
            // Destroy the SwigluDescriptor
            destroyMatmulDescriptor(descriptor);
            destroyTensorDescriptor(c.layout);
            destroyTensorDescriptor(a.layout);
            destroyTensorDescriptor(b.layout);
        }
    }

    fn reform<T, U>(&self, dst: &mut rustTensor<T>, src: &rustTensor<U>, stream: *mut ::std::os::raw::c_void)
    where
        T: DerefMut<Target = [u8]>,
        U: Deref<Target = [u8]>,
    {
        let device = DeviceEnum::DevCpu;
        let config: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let descriptor = createReformDescriptor(device, config) as *mut ReformDescriptor;
    
            let dst = to_ctensor(dst);
            let src = to_ctensor(src);
    
            reform(descriptor, dst, src, stream);
    
            // Destroy the SwigluDescriptor
            destroyReformDescriptor(descriptor);
            destroyTensorDescriptor(dst.layout);
            destroyTensorDescriptor(src.layout);
        }
    }

    fn softmax<T>(&self, att: &mut rustTensor<T>, stream: *mut ::std::os::raw::c_void)
    where
        T: DerefMut<Target = [u8]>,
    {
        let device = DeviceEnum::DevCpu;
        let config: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let descriptor = createCausalSoftmaxDescriptor(device, config) as *mut CausalSoftmaxDescriptor;
    
            let att = to_ctensor(att);
    
            causalSoftmax(descriptor, att, stream);
    
            // Destroy the SwigluDescriptor
            destroyCausalSoftmaxDescriptor(descriptor);
            destroyTensorDescriptor(att.layout);
        }
    }

    fn swiglu<T, U>(&self, gate: &mut rustTensor<T>, up: &rustTensor<U>, stream: *mut ::std::os::raw::c_void)
    where
        T: DerefMut<Target = [u8]>,
        U: Deref<Target = [u8]>,
    {
        let device = DeviceEnum::DevCpu;
        let config: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            let descriptor = createSwigluDescriptor(device, config) as *mut SwigluDescriptor;
    
            let gate = to_ctensor(gate);
            let up = to_ctensor(up);
    
            swiglu(descriptor, gate, up, stream);
    
            // Destroy the SwigluDescriptor
            destroySwigluDescriptor(descriptor);
            destroyTensorDescriptor(gate.layout);
            destroyTensorDescriptor(up.layout);
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
    pub fn sprout<'ctx>(&mut self, ctx: &'ctx ContextGuard) -> <T as ContextSpore>::Resource<'ctx> {
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
