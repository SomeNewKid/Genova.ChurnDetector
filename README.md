# Genova.ChurnDetector

A .NET 8 churn-signal detector that classifies short text as churn-related or not churn-related.

> [!WARNING]
> This is an experimental project and should not be considered production-ready. It exists to explore a small AI, ML, agent, or demo idea within the broader Genova ecosystem.

> [!IMPORTANT]
> A fresh public clone of this repository should not be expected to restore or build without additional Genova infrastructure. Many Genova dependencies are distributed through a private authenticated NuGet feed, and the public source does not include feed credentials or a complete public package graph.

## Installation

```bash
dotnet restore
dotnet build
```

## Usage

Run the terminal app:

```bash
dotnet run --project ChurnDetector.Terminal
```

Train and write runtime artifacts:

```bash
dotnet run --project ChurnDetector.Training
```

## Features

* Detects churn-related language in text
* Exposes a `Detector` API for scoring and explanation
* Includes a terminal app for interactive or automated evaluation
* Includes a training app that generates runtime artifacts from CSV data

## Notes

* Targets .NET 8.
* The terminal app can use `GENOVA_TRAIN_CSV` to locate the training CSV for automated evaluation.
* The training project expects `Input/training.seed.csv` and writes model artifacts used by the library at runtime.

## Thanks

* [ML.NET](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet)

## Third-Party Notices

This project has direct runtime dependencies on third-party NuGet packages, including `Microsoft.Extensions.*` packages (MIT), `Microsoft.ML*` packages (MIT). See each package's NuGet license metadata for full license and notice terms.

## License

GNU General Public License v3.0. See the `LICENSE` file for details.
