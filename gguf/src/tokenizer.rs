use super::GGufModel;
use ggus::{GGmlTokenType, GGufMetaMapExt};
use std::{
    borrow::Cow,
    collections::HashMap,
    str::{from_utf8, from_utf8_unchecked},
};
use tokeneer::{utok, Bpe, Lpe, Method, Tokeneer};
use tokenizers::tokenizer::Tokenizer as Hf;

pub struct Tokenizer {
    tokenize: Box<dyn Tokenize>,
    en_replace: HashMap<char, char>,
    de_replace: HashMap<char, char>,
    hf: Option<Hf>,
}

impl GGufModel<'_> {
    pub fn tokenizer(&self) -> Tokenizer {
        if let Ok("deepseek-r1-qwen") = self.get_str("tokenizer.ggml.pre") {
            return Tokenizer::deepseek(self);
        }
        match self.tokenizer_ggml_model().unwrap() {
            "llama" => Tokenizer::bpe_from_gguf(self),
            "fm9g8b" | "gpt2" => Tokenizer::lpe_from_gguf(self),
            model => panic!("Unsupported tokenizer model: {model}"),
        }
    }
}

impl Tokenizer {
    pub fn encode(&self, text: &str) -> Vec<utok> {
        if let Some(hf) = &self.hf {
            let x = hf.encode(text, false).unwrap();
            return x.get_ids().to_vec();
        }

        let space = self.en_replace[&' '];
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
            text.push(*self.en_replace.get(&c).unwrap_or(&c))
        }
        self.tokenize.encode(&text)
    }

    pub fn decode(&self, token: utok) -> Cow<str> {
        if let Some(hf) = &self.hf {
            let x = hf.decode(&[token], false).unwrap();
            return x.into();
        }

        let piece = self.tokenize.decode(token);
        if let Ok(piece) = from_utf8(piece) {
            let ans = piece
                .chars()
                .map(|c| *self.de_replace.get(&c).unwrap_or(&c))
                .collect::<String>();
            if ans == piece {
                piece.into()
            } else {
                ans.into()
            }
        } else {
            unsafe { from_utf8_unchecked(piece) }.into()
        }
    }

    fn bpe_from_gguf(gguf: &GGufModel) -> Self {
        let tokens = gguf.tokenizer_ggml_tokens().unwrap();
        let scores = gguf.tokenizer_ggml_scores().unwrap();
        let token_type = gguf.tokenizer_ggml_token_type().unwrap();
        assert_eq!(tokens.len(), scores.len());
        assert_eq!(tokens.len(), token_type.len());

        let mut detective = SpaceDetective::new();
        let vocabs = tokens.map(|piece| {
            let piece = piece.unwrap();
            detective.record(piece);
            piece
        });
        let scores = scores.map(|score| score.unwrap());
        let is_byte = token_type.map(|ty| GGmlTokenType::Byte as i32 == ty.unwrap());

        let unk = gguf.tokenizer_ggml_unknown_token_id().unwrap();
        let bos = gguf.tokenizer_ggml_bos_token_id().unwrap();
        let eos = gguf.tokenizer_ggml_eos_token_id().unwrap();

        let bpe = Bpe::new(vocabs, scores, is_byte, unk);
        let bos_piece = from_utf8(bpe.decode(bos)).unwrap().to_string();
        let eos_piece = from_utf8(bpe.decode(eos)).unwrap().to_string();

        let mut tokeneer = Tokeneer::new(bpe);
        tokeneer.extend_special([(bos_piece, vec![bos]), (eos_piece, vec![eos])]);

        let (en_replace, de_replace) = detective.build_map();
        Self {
            tokenize: Box::new(tokeneer),
            en_replace,
            de_replace,
            hf: None,
        }
    }

    fn lpe_from_gguf(gguf: &GGufModel) -> Self {
        let tokens = gguf.tokenizer_ggml_tokens().unwrap();

        let mut detective = SpaceDetective::new();
        let vocabs = tokens.map(|piece| {
            let piece = piece.unwrap();
            detective.record(piece);
            piece.as_bytes()
        });

        let bos = gguf.tokenizer_ggml_bos_token_id().unwrap();
        let eos = gguf.tokenizer_ggml_eos_token_id().unwrap();
        let unk = gguf
            .tokenizer_ggml_unknown_token_id()
            .or(gguf.tokenizer_ggml_padding_token_id())
            .unwrap_or_else(|_| {
                assert_eq!(bos, eos);
                bos
            });

        let bpe = Lpe::new(vocabs, unk);
        let bos_piece = from_utf8(bpe.decode(bos)).unwrap().to_string();
        let eos_piece = from_utf8(bpe.decode(eos)).unwrap().to_string();

        let mut tokeneer = Tokeneer::new(bpe);
        tokeneer.extend_special([(bos_piece, vec![bos]), (eos_piece, vec![eos])]);

        let (en_replace, de_replace) = detective.build_map();
        Self {
            tokenize: Box::new(tokeneer),
            en_replace,
            de_replace,
            hf: None,
        }
    }

    fn deepseek(gguf: &GGufModel) -> Self {
        let mut ans = Tokenizer::lpe_from_gguf(gguf);
        ans.hf =
            Some(Hf::from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", None).unwrap());
        ans
    }
}

/// A trait for tokenization.
trait Tokenize {
    /// Encode a text into a sequence of tokens.
    fn encode(&self, text: &str) -> Vec<utok>;
    /// Decode a token into str.
    fn decode(&self, token: utok) -> &[u8];
}

impl<M: tokeneer::Method> Tokenize for Tokeneer<M> {
    #[inline]
    fn encode(&self, text: &str) -> Vec<utok> {
        self.encode(text)
    }
    #[inline]
    fn decode(&self, token: utok) -> &[u8] {
        self.internal().decode(token)
    }
}

struct SpaceDetective([Record; 3]);
const SPACE: char = ' ';
const SPACE_: char = '▁';
const SPACEG: char = 'Ġ';

struct Record {
    c: char,
    maybe: bool,
    count: usize,
}

impl Record {
    fn new(c: char) -> Self {
        Self {
            c,
            maybe: false,
            count: 0,
        }
    }
}

impl SpaceDetective {
    fn new() -> Self {
        Self([SPACE, SPACE_, SPACEG].map(Record::new))
    }

    fn record(&mut self, text: &str) {
        let mut buf = [0u8; 4];
        for record in self.0.iter_mut() {
            if text == record.c.encode_utf8(&mut buf) {
                record.maybe = true;
                record.count += 1;
                break;
            }
            if text.contains(record.c) {
                record.count += 1
            }
        }
    }

    fn build_map(&self) -> (HashMap<char, char>, HashMap<char, char>) {
        let replace = self
            .0
            .iter()
            .filter(|r| r.maybe)
            .max_by_key(|r| r.count)
            .unwrap()
            .c;
        let en = match replace {
            SPACE => HashMap::from([(SPACE, SPACE)]),
            SPACE_ => HashMap::from([(SPACE, SPACE_)]),
            SPACEG => HashMap::from([(SPACE, SPACEG), ('\n', 'Ċ')]),
            _ => unreachable!(),
        };
        let de = en.iter().map(|(&k, &v)| (v, k)).collect();
        (en, de)
    }
}

#[test]
fn test_load() {
    use test_utils::Inference;
    let Some(Inference { model, prompt, .. }) = Inference::load() else {
        return;
    };
    let gguf = GGufModel::read(model.iter().map(|s| &**s));
    let tokenizer = gguf.tokenizer();
    println!("{:?}", tokenizer.encode(&prompt));
}
