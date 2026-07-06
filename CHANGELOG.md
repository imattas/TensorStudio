# Changelog

## 0.1.1

- Publishable wheel release for CPython 3.10-3.13.
- Fixed MSVC build portability by removing non-portable `ssize_t` usage.
- Updated wheel CI to build 64-bit Windows wheels and modern Linux wheels.
- Expanded documentation with MkDocs navigation and detailed usage guides.

## 0.1.0

- Initial experimental release.
- C++20 CPU tensor core with Python bindings.
- Basic broadcasting, matrix multiplication, reductions, activations, and views.
- Reverse-mode autograd for the initial operation set.
- Python `nn`, `optim`, serialization, examples, tests, docs, and CI workflows.
