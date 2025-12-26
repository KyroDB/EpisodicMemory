"""
Test suite for Vritti episodic memory system.

This package contains tests organized by type:
    - Unit tests: Individual component testing (test_*.py in root)
    - Integration tests: End-to-end workflow testing (integration/)
    - Load tests: Performance and scalability testing (load/)
    - Chaos tests: Fault injection and resilience testing (chaos/)

Test markers:
    - unit: Fast unit tests
    - integration: Tests requiring external services (KyroDB, LLM)
    - slow: Tests taking >1 second
    - requires_kyrodb: Tests requiring KyroDB connection
    - load: Load and performance tests

Run tests:
    pytest                    # All tests
    pytest -m unit            # Unit tests only
    pytest -m "not slow"      # Skip slow tests
    pytest tests/integration  # Integration tests only
"""
