use digit_layout::DigitLayout;
use ggus::{GGuf, GGufError, GGufFileName, GGufMetaKV, GENERAL_ALIGNMENT};
use memmap2::Mmap;
use std::{collections::HashMap, fmt::Debug, fs::File, path::Path};

/// 从指定文件的路径出发，映射所有分片文件。
pub fn map_files(path: impl AsRef<Path>) -> Box<[Mmap]> {
    fn throw(path: &Path, e: impl Debug) -> ! {
        let path = path.display();
        panic!(
            "\
Error occurred at path: {path}
  error: {e:?}"
        )
    }

    #[inline]
    fn map_file(path: &Path) -> Mmap {
        let file = File::open(path).unwrap_or_else(|e| throw(path, e));
        unsafe { Mmap::map(&file) }.unwrap()
    }

    let path = path.as_ref();
    let name = GGufFileName::try_from(path).unwrap_or_else(|e| throw(path, e));

    if name.shard_count() == 1 {
        Box::new([map_file(path)])
    } else {
        let dir = path.parent().unwrap();
        name.iter_all()
            .map(|name| map_file(&dir.join(name.to_string())))
            .collect()
    }
}

/// GGuf 模型，可能来自多个分片文件。
pub struct GGufModel<'a> {
    /// 元数据键值对。
    pub meta_kvs: HashMap<&'a str, GGufMetaKV<'a>>,
    /// 张量。
    pub tensors: HashMap<&'a str, GGufTensor<'a>>,
}

/// GGuf 张量。
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct GGufTensor<'a> {
    pub ty: DigitLayout,
    pub shape: Box<[usize]>,
    pub data: &'a [u8],
}

impl<'a> GGufModel<'a> {
    /// 从多个分片文件中读取 GGuf 模型。
    pub fn read(files: impl IntoIterator<Item = &'a [u8]> + 'a) -> Self {
        let mut ans = Self {
            meta_kvs: Default::default(),
            tensors: Default::default(),
        };
        std::thread::scope(|s| {
            for (i, thread) in files
                .into_iter()
                .map(|data| s.spawn(|| GGuf::new(data)))
                .collect::<Vec<_>>()
                .into_iter()
                .enumerate()
            {
                thread
                    .join()
                    .unwrap()
                    .and_then(|gguf| ans.merge(gguf))
                    .unwrap_or_else(|e| panic!("Error at file {i}: {e}"));
            }
        });
        ans
    }

    fn merge(&mut self, others: GGuf<'a>) -> Result<(), GGufError> {
        for (k, kv) in others.meta_kvs {
            if k == GENERAL_ALIGNMENT || k.starts_with("split.") {
                continue;
            }
            if self.meta_kvs.insert(k, kv).is_some() {
                return Err(GGufError::DuplicateMetaKey(k.into()));
            }
        }

        for (name, t) in others.tensors {
            use digit_layout::types as ty;
            use ggus::GGmlType as Ty;

            let t = t.to_info();
            let t = GGufTensor {
                ty: match t.ty() {
                    Ty::F32 => ty::F32,
                    Ty::F16 => ty::F16,
                    Ty::I8 => ty::I8,
                    Ty::I16 => ty::I16,
                    Ty::I32 => ty::I32,
                    Ty::I64 => ty::I64,
                    Ty::F64 => ty::F64,
                    _ => todo!(),
                },
                // gguf 张量的形状存储方式与一般情况相反
                shape: t.shape().iter().rev().map(|&d| d as _).collect(),
                data: &others.data[t.offset()..][..t.nbytes()],
            };

            if self.tensors.insert(name, t).is_some() {
                return Err(GGufError::DuplicateTensorName(name.into()));
            }
        }

        Ok(())
    }
}
