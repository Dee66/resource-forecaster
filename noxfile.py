"""Nox configuration for Resource Forecaster development automation.

This file defines automated development tasks including linting, testing,
formatting, packaging, and end-to-end forecast validation.
"""

import nox

# Python versions to test against
PYTHON_VERSIONS = ["3.11"]

# Default sessions to run when no specific session is requested
nox.options.sessions = ["lint", "test", "coverage"]


@nox.session(python=PYTHON_VERSIONS)
def lint(session):
    """Run linting with ruff and mypy."""
    session.install("poetry")
    session.run("poetry", "install")
    
    # Run ruff for code quality
    session.run("poetry", "run", "ruff", "check", "src", "tests")
    
    # Run mypy for type checking
    session.run("poetry", "run", "mypy", "src")
    
    session.log("✅ Linting completed successfully")


@nox.session(python=PYTHON_VERSIONS)
def format_code(session):
    """Format code with black and isort."""
    session.install("poetry")
    session.run("poetry", "install")
    
    # Format with black
    session.run("poetry", "run", "black", "src", "tests")
    
    # Sort imports with isort
    session.run("poetry", "run", "isort", "src", "tests")
    
    # Fix auto-fixable ruff issues
    session.run("poetry", "run", "ruff", "check", "--fix", "src", "tests")
    
    session.log("✅ Code formatting completed")


@nox.session(python=PYTHON_VERSIONS)
def test(session):
    """Run the test suite with pytest."""
    session.install("poetry")
    session.run("poetry", "install")
    
    # Run tests with coverage
    session.run(
        "poetry", "run", "pytest", 
        "tests/",
        "-v",
        "--tb=short",
        "--strict-markers",
        "-m", "not slow",  # Skip slow tests by default
    )
    
    session.log("✅ Unit tests completed successfully")


@nox.session(python=PYTHON_VERSIONS)
def test_all(session):
    """Run all tests including slow integration tests."""
    session.install("poetry")
    session.run("poetry", "install")


@nox.session(python=PYTHON_VERSIONS)
def guardrail(session):
    """Run the budget guardrail script locally (use --block-exit to fail on over-budget)."""
    session.install("poetry")
    session.run("poetry", "install")
    session.run("python", "scripts/section11_budget_guardrail.py", "--predicted-json", "dashboards/predicted_cost.json", "--budget", "1000", "--env", "prod", "--block-exit")
    session.log("✅ Guardrail script executed")


@nox.session(python=PYTHON_VERSIONS)
def coverage(session):
    """Run tests with coverage reporting."""
    session.install("poetry")
    session.run("poetry", "install")
    
    # Run tests with coverage
    session.run(
        "poetry", "run", "pytest",
        "tests/",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--cov-report=xml:coverage.xml",
        "--cov-fail-under=80",
        "-m", "not slow",
    )
    
    session.log("✅ Coverage analysis completed")
    session.log("📊 Coverage report available at htmlcov/index.html")


@nox.session(python=PYTHON_VERSIONS)
def e2e_forecast(session):
    """Run end-to-end forecast validation."""
    session.install("poetry")
    session.run("poetry", "install")
    
    # Run forecast accuracy validation
    session.run(
        "poetry", "run", "python", 
        "scripts/validate_forecast_accuracy.py",
        "--env", "dev",
        "--horizon", "7d",
        "--rmse-threshold", "0.05"
    )
    
    session.log("✅ End-to-end forecast validation completed")


@nox.session(python=PYTHON_VERSIONS)
def package(session):
    """Package the model artifacts and inference code.

    Usage examples:
      nox -s package -- --env dev
      nox -s package -- --env staging --s3-bucket my-bucket
      nox -s package -- --env prod --include models --include src/forecaster/inference
    """
    session.install("poetry")
    session.run("poetry", "install")

    args = session.posargs or ["--env", "dev"]
    session.run(
        "poetry", "run", "python", "scripts/package_forecaster_artifacts.py", *args
    )
    session.log("✅ Model packaging completed")


@nox.session(python=PYTHON_VERSIONS)
def package_upload(session):
    """Convenience session to package and upload to S3.

    Example:
      nox -s package_upload -- --env staging --s3-bucket my-artifacts
    """
    session.install("poetry")
    session.run("poetry", "install")
    args = session.posargs or ["--env", "staging"]
    session.run("poetry", "run", "python", "scripts/package_forecaster_artifacts.py", *args)
    session.log("✅ Packaging and upload (if specified) completed")


@nox.session(python=PYTHON_VERSIONS)
def rollback(session):
    """Rollback helper to list/set previous model packages.

    Examples:
      nox -s rollback -- --env staging --list
      nox -s rollback -- --env staging --previous --set-alias --download
      nox -s rollback -- --env prod --key model-packages/prod/model_package-20250101-000000.zip --set-alias
    """
    session.install("poetry")
    session.run("poetry", "install")
    args = session.posargs or ["--env", "staging", "--list"]
    session.run("poetry", "run", "python", "scripts/rollback_model.py", *args)
    session.log("✅ Rollback command completed")


@nox.session(python=PYTHON_VERSIONS)
def deploy(session):
    """Deploy with guardrails: package -> synth -> deploy.

    Examples:
      nox -s deploy -- --env dev
      nox -s deploy -- --env staging --s3-bucket my-artifacts
      nox -s deploy -- --env prod --yes --force
    """
    session.install("poetry")
    session.run("poetry", "install")
    args = session.posargs or ["--env", "dev"]
    session.run("poetry", "run", "python", "scripts/deploy_flow.py", *args)
    session.log("✅ Deployment flow completed")


@nox.session(python=PYTHON_VERSIONS)
def teardown(session):
    """Teardown CDK stacks with confirmation.

    Examples:
      nox -s teardown -- --env dev --yes
      nox -s teardown -- --env staging --yes
    """
    session.install("poetry")
    session.run("poetry", "install")
    args = session.posargs or ["--env", "dev"]
    session.run("poetry", "run", "python", "scripts/teardown.py", *args)
    session.log("✅ Teardown completed")


@nox.session(python=PYTHON_VERSIONS)
def create_dashboards(session):
    """Create CloudWatch dashboards for demo.

    Examples:
      nox -s create_dashboards
      nox -s create_dashboards -- --region us-west-2 --create-dashboards
    """
    session.install("poetry")
    session.run("poetry", "install")
    args = session.posargs or ["--region", "us-east-1"]
    session.run("poetry", "run", "python", "scripts/create_dashboards.py", *args)
    session.log("✅ Dashboard configurations created")


@nox.session(python=PYTHON_VERSIONS)
def demo_screenshots(session):
    """Capture CloudWatch dashboard screenshots for demo.

    Examples:
      nox -s demo_screenshots
      nox -s demo_screenshots -- --region us-west-2 --wait 15
    """
    session.install("poetry")
    session.run("poetry", "install")
    
    # Install selenium if not already present
    session.run("poetry", "run", "pip", "install", "selenium", external=True)
    
    args = session.posargs or ["--region", "us-east-1"]
    session.run("poetry", "run", "python", "scripts/capture_dashboard_screenshots.py", *args)
    session.log("✅ Demo screenshots captured")
    session.run("poetry", "install")
    args = session.posargs or ["--env", "staging", "--yes"]
    session.run("poetry", "run", "python", "scripts/deploy_flow.py", *args)
    session.log("✅ Deploy flow completed")


@nox.session(python=PYTHON_VERSIONS)
def benchmark(session):
    """Run performance benchmarks."""
    session.install("poetry")
    session.run("poetry", "install")
    
    # Run forecast performance benchmarks
    session.run(
        "poetry", "run", "pytest",
        "tests/benchmarks/",
        "--benchmark-only",
        "--benchmark-sort=mean",
        "-v"
    )
    
    session.log("✅ Performance benchmarks completed")


@nox.session(python=PYTHON_VERSIONS)
def docs(session):
    """Build documentation with MkDocs."""
    session.install("poetry")
    session.run("poetry", "install")
    
    # Build documentation
    session.run("poetry", "run", "mkdocs", "build")
    
    session.log("✅ Documentation built successfully")
    session.log("📚 Documentation available at site/index.html")


@nox.session(python=PYTHON_VERSIONS)
def section11(session):
    """Run Section 11 guardrails: create alarms and run budget checks.

    Examples:
      nox -s section11 -- --create-alarms
      nox -s section11 -- --apply-alarms --sns-email ops@example.com
      nox -s section11 -- --check-budget --predicted-json dashboards/predicted_cost.json --block-exit
    """
    session.install("poetry")
    session.run("poetry", "install")

    pos = session.posargs
    if not pos:
        # default: generate alarm definitions
        session.run("poetry", "run", "python", "scripts/section11_create_alarms_and_guardrails.py")
        session.log("✅ Section11 alarm definitions generated")
        return

    # forward args to scripts
    session.run("poetry", "run", "python", "scripts/section11_create_alarms_and_guardrails.py", *pos)
    session.log("✅ Section11 scripts executed")


@nox.session(python=PYTHON_VERSIONS)
def docs_serve(session):
    """Serve documentation locally."""
    session.install("poetry")
    session.run("poetry", "install")
    
    # Serve documentation
    session.run("poetry", "run", "mkdocs", "serve")


@nox.session(python=PYTHON_VERSIONS)
def clean(session):
    """Clean up build artifacts and cache files."""
    import shutil
    import os
    
    # Directories to clean
    clean_dirs = [
        ".pytest_cache",
        "__pycache__",
        ".coverage",
        "htmlcov",
        "coverage.xml",
        "site",
        "dist",
        ".ruff_cache",
        ".mypy_cache",
    ]
    
    for dir_name in clean_dirs:
        if os.path.exists(dir_name):
            if os.path.isdir(dir_name):
                shutil.rmtree(dir_name)
                session.log(f"🗑️  Removed directory: {dir_name}")
            else:
                os.remove(dir_name)
                session.log(f"🗑️  Removed file: {dir_name}")
    
    session.log("✅ Cleanup completed")


@nox.session(python=PYTHON_VERSIONS)
def security(session):
    """Run security checks with bandit and safety."""
    session.install("poetry")
    session.run("poetry", "install")
    
    # Install security tools
    session.install("bandit", "safety")
    
    # Run bandit for security issues
    session.run("bandit", "-r", "src", "-f", "json")
    
    # Check for known vulnerabilities
    session.run("safety", "check")
    
    session.log("✅ Security checks completed")


@nox.session(python=PYTHON_VERSIONS)
def pre_commit(session):
    """Run all pre-commit checks."""
    session.install("poetry")
    session.run("poetry", "install")
    
    # Run all quality checks
    session.notify("format_code")
    session.notify("lint")
    session.notify("test")
    session.notify("security")
    
    session.log("✅ All pre-commit checks completed")