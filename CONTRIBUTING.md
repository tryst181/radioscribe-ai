# Contributing to RadioScribe AI

Thank you for your interest in contributing! This project is a medical AI system, so contributions must adhere to strict quality and safety standards.

## Development Setup

1. Fork the repository
2. Create a virtual environment: `python -m venv venv`
3. Install dependencies: `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and configure
5. Run tests: `pytest tests/ -v`

## Code Standards

- **Type hints** are mandatory for all function signatures
- **Docstrings** required for all public methods (Google style)
- **Safety-critical code** (vision, reasoning) requires review from a maintainer
- Follow PEP 8 style guidelines

## Pull Request Process

1. Create a feature branch from `main`
2. Add tests for any new functionality
3. Ensure all tests pass (`pytest tests/ -v`)
4. Update documentation if API surface changes
5. Submit PR with a clear description of changes

## Safety Guidelines

- Never remove or weaken hedging language in the reasoning system prompt
- Never disable blocking explainability checks
- All model changes must include accuracy benchmarks
- Clinical text generation changes require medical review

## Reporting Issues

Use GitHub Issues with the appropriate template. Include:
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, GPU)
