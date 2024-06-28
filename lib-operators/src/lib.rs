// Include the bindings
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
extern crate libc; // 使用 libc crate 提供一些 C 标准库类型和函数
use digit_layout::types::F16 as RF16;

use tensor::{Tensor as rustTensor, reslice};
use digit_layout::DigitLayout;
use std::ops::Deref;

use common::{Blob, f16};

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

#[test]
fn test_import() {

    // Create a device (using the enum from the bindings)
    let device = DeviceEnum::DevCpu;

    // Example configuration (replace with actual config data as needed)
    let config: *mut std::ffi::c_void = std::ptr::null_mut();


    unsafe {
        let descriptor = createSwigluDescriptor(device, config) as *mut SwigluDescriptor;
        if descriptor.is_null() {
            eprintln!("Failed to create descriptor");
            return;
        }

        println!("debug===================================================");
        let rlayout = RF16;
        let mut gate_rtensor = rustTensor::alloc(rlayout, &[1, 64], Blob::new);
        let gate_data = gate_rtensor.physical_mut();

        let arr = [1.0; 64];
        let src = &arr
            .iter()
            .map(|x| f16::from_f32(*x as f32))
            .collect::<Vec<_>>();
        gate_data.copy_from_slice(reslice(&src));        

        let mut up_rtensor = rustTensor::alloc(rlayout, &[1, 64], Blob::new);
        let up_data = up_rtensor.physical_mut();
        up_data.copy_from_slice(reslice(&src));

        let gate = to_ctensor(&gate_rtensor);


        let up = to_ctensor(&up_rtensor);

        println!("gate: {:?}, up: {:?}", gate, up);

        // Example stream (replace with actual stream)
        let stream: *mut std::ffi::c_void = std::ptr::null_mut();

        // Call the swiglu function
        swiglu(descriptor, gate, up, stream);

        println!("gate: {}\n", gate_rtensor);

        // Destroy the SwigluDescriptor
        destroySwigluDescriptor(descriptor);
        destroyTensorDescriptor(gate.layout);
        destroyTensorDescriptor(up.layout);
    }
}
