use hyper::{Method, StatusCode};

#[derive(Debug)]
pub(crate) enum Error {
    WrongJson(serde_json::Error),
    NotFound(NotFoundError),
}

#[derive(serde::Serialize, Debug)]
pub(crate) struct NotFoundError {
    method: String,
    uri: String,
}

impl Error {
    pub fn not_found(method: &Method, uri: &str) -> Self {
        Self::NotFound(NotFoundError {
            method: method.to_string(),
            uri: uri.into(),
        })
    }

    #[inline]
    pub const fn status(&self) -> StatusCode {
        match self {
            Self::WrongJson(..) => StatusCode::BAD_REQUEST,
            Self::NotFound(..) => StatusCode::NOT_FOUND,
        }
    }

    #[inline]
    pub fn body(&self) -> String {
        match self {
            Self::WrongJson(e) => e.to_string(),
            Self::NotFound(e) => serde_json::to_string(&e).unwrap(),
        }
    }
}
