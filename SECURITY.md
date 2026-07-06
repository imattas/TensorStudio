# Security Policy

TensorStudio `1.0.0rc1` is a CPU-only release candidate.

## Reporting a Vulnerability

Please report suspected vulnerabilities by opening a private security advisory
on GitHub or by contacting the maintainers through the project repository.

Include:

- Affected version or commit
- Reproduction steps
- Impact assessment
- Suggested fix, if available

## Serialization Warning

`tensorstudio.load()` uses Python pickle for internal object roundtrips.
Loading pickle files from untrusted sources is unsafe because pickle can execute
arbitrary code during deserialization.
