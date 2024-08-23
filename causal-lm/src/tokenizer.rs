use common::GGufModel;
use ggus::{GGmlTokenType, GGufMetaDataValueType};
use std::{
    borrow::Cow,
    str::{from_utf8, from_utf8_unchecked},
};
use tokeneer::{utok, Bpe, Method, Tokeneer};

pub struct Tokenizer {
    tokenize: Box<dyn Tokenize>,
    replace_space: Option<char>,
}

impl Tokenizer {
    pub fn from_gguf(gguf: &GGufModel) -> Self {
        let model = gguf.meta_kvs["tokenizer.ggml.model"]
            .value_reader()
            .read_str()
            .unwrap();
        match model {
            "llama" => Self::bpe_from_gguf(gguf),
            _ => panic!("Unsupported tokenizer model: {model}"),
        }
    }

    pub fn encode(&self, text: &str) -> Vec<utok> {
        let space = self.replace_space.unwrap_or(' ');
        let mut chars = text.chars();
        let mut text = match chars.next() {
            Some(c) => {
                if c.is_ascii_alphabetic() {
                    format!("{space}{c}")
                } else {
                    format!("{c}")
                }
            }
            None => return vec![],
        };
        for c in chars {
            text.push(match c {
                ' ' => space,
                c => c,
            })
        }
        self.tokenize.encode(&text)
    }
    pub fn decode(&self, token: utok) -> Cow<str> {
        let piece = self.tokenize.decode(token);
        if let Some(c) = self.replace_space {
            piece.replace(c, " ").into()
        } else {
            piece.into()
        }
    }

    fn bpe_from_gguf(gguf: &GGufModel) -> Self {
        let _pre = gguf.meta_kvs["tokenizer.ggml.pre"]
            .value_reader()
            .read_str()
            .unwrap();
        let mut tokens = gguf.meta_kvs["tokenizer.ggml.tokens"].value_reader();
        let mut scores = gguf.meta_kvs["tokenizer.ggml.scores"].value_reader();
        let mut token_type = gguf.meta_kvs["tokenizer.ggml.token_type"].value_reader();

        let unk = gguf.meta_kvs["tokenizer.ggml.unknown_token_id"]
            .value_reader()
            .read::<utok>()
            .unwrap();
        let bos = gguf.meta_kvs["tokenizer.ggml.bos_token_id"]
            .value_reader()
            .read::<utok>()
            .unwrap();
        let eos = gguf.meta_kvs["tokenizer.ggml.eos_token_id"]
            .value_reader()
            .read::<utok>()
            .unwrap();

        let (ty, len) = tokens.read_arr_header().unwrap();
        assert_eq!(ty, GGufMetaDataValueType::String);

        let (ty, len_) = scores.read_arr_header().unwrap();
        assert_eq!(ty, GGufMetaDataValueType::F32);
        assert_eq!(len_, len);

        let (ty, len_) = token_type.read_arr_header().unwrap();
        assert_eq!(ty, GGufMetaDataValueType::I32);
        assert_eq!(len_, len);
        //
        let mut space_exist = false;
        let mut replace_exist = false;
        let vocabs = (0..len).map(|_| {
            let piece = tokens.read_str().unwrap();
            match piece {
                " " => space_exist = true,
                "▁" => replace_exist = true,
                _ => {}
            }
            piece
        });
        let scores = (0..len).map(|_| scores.read::<f32>().unwrap());
        let is_byte = (0..len).map(|_| GGmlTokenType::Byte == token_type.read().unwrap());

        let bpe = Bpe::new(vocabs, scores, is_byte, unk);
        let bos_piece = from_utf8(bpe.decode(bos)).unwrap().to_string();
        let eos_piece = from_utf8(bpe.decode(eos)).unwrap().to_string();

        let mut tokeneer = Tokeneer::new(bpe);
        tokeneer.extend_special([(bos_piece, vec![bos]), (eos_piece, vec![eos])]);
        Self {
            tokenize: Box::new(tokeneer),
            replace_space: match (space_exist, replace_exist) {
                (true, _) => None,
                (false, true) => Some('▁'),
                (false, false) => panic!("Unknown user-defined space"),
            },
        }
    }
}

/// A trait for tokenization.
trait Tokenize {
    /// Encode a text into a sequence of tokens.
    fn encode(&self, text: &str) -> Vec<utok>;
    /// Decode a token into str.
    fn decode(&self, token: utok) -> &str;
}

impl<M: tokeneer::Method> Tokenize for Tokeneer<M> {
    #[inline]
    fn encode(&self, text: &str) -> Vec<utok> {
        self.encode(text)
    }
    #[inline]
    fn decode(&self, token: utok) -> &str {
        unsafe { from_utf8_unchecked(self.internal().decode(token)) }
    }
}
