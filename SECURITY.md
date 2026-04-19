# Security Policy

## Supported Versions

We actively maintain security fixes for the latest release. Older versions do not receive backported patches.

| Version | Supported |
| ------- | --------- |
| latest  | ✓         |

## Reporting a Vulnerability

If you discover a security vulnerability in `react-native-executorch`, **please do not open a public GitHub issue.**

Instead, email **security@bedda.tech** with:

- A description of the vulnerability and its potential impact
- Steps to reproduce (proof-of-concept code is welcome)
- The versions you have tested against

We will acknowledge your report within **48 hours** and aim to release a fix within **14 days** of confirmation.

We do not currently offer a bug bounty program, but we will credit researchers in the release notes unless they request anonymity.

## Security Considerations

`react-native-executorch` runs machine learning models on-device. Integrators should be aware of:

- **Model provenance**: only load `.pte` model files from trusted sources. A maliciously crafted model file could exploit parser vulnerabilities in the ExecuTorch runtime.
- **Prompt injection**: LLMs may be susceptible to adversarial inputs embedded in user-supplied text. Sanitize or validate prompts before passing them to the model, especially in agent contexts where the model output drives real actions.
- **Resource exhaustion**: on-device inference is CPU/GPU intensive. Implement timeouts and cancellation to prevent denial-of-service from runaway inference loops.
