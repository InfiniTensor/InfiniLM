//! All HttpResponses in this App.

use super::error::Error;
use http_body_util::{BodyExt, Full, StreamBody, combinators::BoxBody};
use hyper::{
    Response, StatusCode,
    body::{Bytes, Frame},
    header::{
        ACCESS_CONTROL_ALLOW_HEADERS, ACCESS_CONTROL_ALLOW_METHODS, ACCESS_CONTROL_ALLOW_ORIGIN,
        CACHE_CONTROL, CONNECTION, CONTENT_TYPE,
    },
};
use serde::Serialize;
use tokio_stream::{Stream, StreamExt};

pub fn text_stream(
    s: impl Stream<Item = String> + Send + Sync + 'static,
) -> Response<BoxBody<Bytes, hyper::Error>> {
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/event-stream")
        .header(CACHE_CONTROL, "no-cache")
        .header(CONNECTION, "keep-alive")
        .header(ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .header(ACCESS_CONTROL_ALLOW_METHODS, "GET,POST")
        .header(ACCESS_CONTROL_ALLOW_HEADERS, "Content-Type")
        .body(StreamBody::new(s.map(|s| Ok(Frame::data(format!("data: {s}\n\n").into())))).boxed())
        .unwrap()
}

pub fn json(json: impl Serialize) -> Response<BoxBody<Bytes, hyper::Error>> {
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "application/json")
        .header(ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .header(ACCESS_CONTROL_ALLOW_METHODS, "GET,POST")
        .header(ACCESS_CONTROL_ALLOW_HEADERS, "Content-Type")
        .body(full(serde_json::to_string(&json).unwrap()))
        .unwrap()
}

pub fn error(e: Error) -> Response<BoxBody<Bytes, hyper::Error>> {
    Response::builder()
        .status(e.status())
        .header(CONTENT_TYPE, "application/json")
        .header(ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .header(ACCESS_CONTROL_ALLOW_METHODS, "GET,POST")
        .header(ACCESS_CONTROL_ALLOW_HEADERS, "Content-Type")
        .body(full(e.body()))
        .unwrap()
}

fn full(chunk: impl Into<Bytes>) -> BoxBody<Bytes, hyper::Error> {
    Full::new(chunk.into())
        .map_err(|never| match never {})
        .boxed()
}
