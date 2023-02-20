[![Crates.io](https://img.shields.io/crates/v/async_once_cell.svg)](https://crates.io/crates/async-once-cell)
[![API reference](https://docs.rs/async-once-cell/badge.svg)](https://docs.rs/async-once-cell/)

# Overview

`async_once_cell` is a version of [once_cell](https://crates.io/crates/once_cell)
that adds support for async initialization of cells. The short version of the
API is:

```rust
impl OnceCell<T> {
    fn new() -> OnceCell<T>;
    fn get(&self) -> Option<&T>;
    async fn get_or_init(&self, init: impl Future<Output=T>) -> &T;
}
```

More patterns and use-cases are in the [docs](https://docs.rs/async_once_cell/)!

# Related crates

* [once_cell](https://crates.io/crates/once_cell)
* [double-checked-cell](https://github.com/niklasf/double-checked-cell)
* [lazy-init](https://crates.io/crates/lazy-init)
* [lazycell](https://crates.io/crates/lazycell)
* [mitochondria](https://crates.io/crates/mitochondria)
* [lazy_static](https://crates.io/crates/lazy_static)
