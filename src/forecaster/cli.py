"""Command-line interface for Resource Forecaster.

Provides CLI commands for training, forecasting, and managing the
FinOps cost prediction system.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import load_config
from .exceptions import ForecasterError

app = typer.Typer(
    name="forecaster",
    help="Resource Forecaster - FinOps & Capacity Planning CLI",
    rich_markup_mode="rich",
)
console = Console()


@app.command()
def train(
    env: str = typer.Option("dev", help="Environment (dev/staging/prod)"),
    config_path: Path | None = typer.Option(None, help="Custom config file path"),
    output_dir: Path | None = typer.Option(None, help="Model output directory"),
    horizon_days: int = typer.Option(30, help="Forecast horizon in days"),
    validate: bool = typer.Option(True, help="Run model validation"),
) -> None:
    """Train the forecasting model with historical cost data."""
    console.print("[bold blue]🏋️  Starting forecaster training...[/bold blue]")

    try:
        # Load configuration
        config = load_config(env, config_path)
        console.print(f"✅ Loaded config for environment: {env}")

        # Import and run training
        from .train.forecaster_train import train_forecaster

        model_path = train_forecaster(
            config=config, output_dir=output_dir, horizon_days=horizon_days, validate=validate
        )

        console.print(
            f"[bold green]✅ Training completed! Model saved to: {model_path}[/bold green]"
        )

    except ForecasterError as e:
        console.print(f"[bold red]❌ Training failed: {e}[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]💥 Unexpected error: {e}[/bold red]")
        sys.exit(1)


@app.command()
def predict(
    env: str = typer.Option("dev", help="Environment (dev/staging/prod)"),
    config_path: Path | None = typer.Option(None, help="Custom config file path"),
    model_path: Path | None = typer.Option(None, help="Path to trained model"),
    horizon_days: int = typer.Option(30, help="Forecast horizon in days"),
    output_path: Path | None = typer.Option(None, help="Output file for predictions"),
) -> None:
    """Generate cost forecasts using the trained model."""
    console.print("[bold blue]🔮 Generating cost forecasts...[/bold blue]")

    try:
        # Load configuration
        config = load_config(env, config_path)
        console.print(f"✅ Loaded config for environment: {env}")

        # Import and run prediction
        from .inference.forecaster_handler import ForecasterHandler

        handler = ForecasterHandler(config, model_path)
        forecasts = handler.predict(horizon_days=horizon_days)

        # Display results
        _display_forecast_summary(forecasts)

        # Save to file if requested
        if output_path:
            forecasts.to_csv(output_path, index=False)
            console.print(f"💾 Forecasts saved to: {output_path}")

        console.print("[bold green]✅ Forecasting completed![/bold green]")

    except ForecasterError as e:
        console.print(f"[bold red]❌ Forecasting failed: {e}[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]💥 Unexpected error: {e}[/bold red]")
        sys.exit(1)


@app.command()
def validate(
    env: str = typer.Option("dev", help="Environment (dev/staging/prod)"),
    config_path: Path | None = typer.Option(None, help="Custom config file path"),
    model_path: Path | None = typer.Option(None, help="Path to trained model"),
    rmse_threshold: float = typer.Option(0.05, help="Maximum allowed RMSE (5%)"),
    days_back: int = typer.Option(90, help="Days of historical data for validation"),
) -> None:
    """Validate forecast model accuracy against historical data."""
    console.print("[bold blue]🔍 Validating forecast accuracy...[/bold blue]")

    try:
        # Load configuration
        config = load_config(env, config_path)
        console.print(f"✅ Loaded config for environment: {env}")

        # Import and run validation
        from .validation.forecast_validator import validate_model_accuracy

        metrics = validate_model_accuracy(
            config=config, model_path=model_path, rmse_threshold=rmse_threshold, days_back=days_back
        )

        # Display validation results
        _display_validation_results(metrics, rmse_threshold)

        if metrics["rmse"] <= rmse_threshold:
            console.print("[bold green]✅ Model validation PASSED![/bold green]")
        else:
            console.print("[bold red]❌ Model validation FAILED![/bold red]")
            sys.exit(1)

    except ForecasterError as e:
        console.print(f"[bold red]❌ Validation failed: {e}[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]💥 Unexpected error: {e}[/bold red]")
        sys.exit(1)


@app.command()
def status(
    env: str = typer.Option("dev", help="Environment (dev/staging/prod)"),
    config_path: Path | None = typer.Option(None, help="Custom config file path"),
) -> None:
    """Show forecaster system status and health metrics."""
    console.print("[bold blue]📊 Checking forecaster status...[/bold blue]")

    try:
        # Load configuration
        config = load_config(env, config_path)

        # Get system status
        from .monitoring.health_check import get_system_status

        status_info = get_system_status(config)

        # Display status table
        _display_status_table(status_info)

    except ForecasterError as e:
        console.print(f"[bold red]❌ Status check failed: {e}[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]💥 Unexpected error: {e}[/bold red]")
        sys.exit(1)


def _display_forecast_summary(forecasts) -> None:
    """Display forecast summary table."""
    table = Table(title="Cost Forecast Summary")
    table.add_column("Date", style="cyan")
    table.add_column("Predicted Cost", style="green")
    table.add_column("Confidence Interval", style="yellow")
    table.add_column("Cost Center", style="blue")

    # Show first 10 rows
    for _, row in forecasts.head(10).iterrows():
        table.add_row(
            str(row["date"]),
            f"${row['predicted_cost']:.2f}",
            f"${row['lower_bound']:.2f} - ${row['upper_bound']:.2f}",
            row.get("cost_center", "N/A"),
        )

    console.print(table)

    if len(forecasts) > 10:
        console.print(f"... and {len(forecasts) - 10} more rows")


def _display_validation_results(metrics: dict, threshold: float) -> None:
    """Display validation results table."""
    table = Table(title="Model Validation Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Status", style="yellow")

    # RMSE
    rmse_status = "✅ PASS" if metrics["rmse"] <= threshold else "❌ FAIL"
    table.add_row("RMSE", f"{metrics['rmse']:.4f}", rmse_status)

    # Other metrics
    table.add_row("MAPE", f"{metrics.get('mape', 0):.2f}%", "")
    table.add_row("R²", f"{metrics.get('r2', 0):.4f}", "")
    table.add_row("Samples", str(metrics.get("samples", 0)), "")

    console.print(table)


def _display_status_table(status: dict) -> None:
    """Display system status table."""
    table = Table(title="Forecaster System Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Details", style="yellow")

    for component, info in status.items():
        status_icon = "✅" if info.get("healthy", False) else "❌"
        table.add_row(
            component.replace("_", " ").title(),
            f"{status_icon} {info.get('status', 'Unknown')}",
            info.get("details", ""),
        )

    console.print(table)


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
