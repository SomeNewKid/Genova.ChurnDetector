# Genova.ChurnDetector

A .NET 8 churn-signal detector that classifies short text as churn-related or not churn-related.

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

## License

GNU General Public License v3.0.
