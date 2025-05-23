//! All HttpResponses in this App.

use super::error::Error;
use http_body_util::{combinators::BoxBody, BodyExt, Full, StreamBody};
use hyper::{
    body::{Bytes, Frame},
    header::{CACHE_CONTROL, CONNECTION, CONTENT_TYPE},
    Response, StatusCode,
};
use tokio_stream::{Stream, StreamExt};

pub fn text_stream(
    s: impl Stream<Item = String> + Send + Sync + 'static,
) -> Response<BoxBody<Bytes, hyper::Error>> {
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/event-stream")
        .header(CACHE_CONTROL, "no-cache")
        .header(CONNECTION, "keep-alive")
        .body(StreamBody::new(s.map(|s| Ok(Frame::data(format!("data: {s}\n\n").into())))).boxed())
        .unwrap()
}

pub fn single(body: String) -> Response<BoxBody<Bytes, hyper::Error>> {
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/event-stream")
        .header(CACHE_CONTROL, "no-cache")
        .header(CONNECTION, "keep-alive")
        .body(full(body))
        .unwrap()
}

pub fn error(e: Error) -> Response<BoxBody<Bytes, hyper::Error>> {
    Response::builder()
        .status(e.status())
        .header(CONTENT_TYPE, "application/json")
        .body(full(e.body()))
        .unwrap()
}

fn full(chunk: impl Into<Bytes>) -> BoxBody<Bytes, hyper::Error> {
    Full::new(chunk.into())
        .map_err(|never| match never {})
        .boxed()
}
