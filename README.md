# Voyagers Forecasting

This project is a readaptation (fork) of the [Chronos](https://github.com/amazon-science/chronos-forecasting) repository, designed for advanced time series forecasting experiments and development.

## Project Structure

The codebase is organized to separate legacy components from the core Chronos-2 implementation and future adaptations:

- **`src/legacy/`**: Contains utility functions and older Chronos modules.
- **`src/chronos2/`**: Contains all pre-existent Chronos-2 modules.

## Notebooks

The `notebooks/` directory contains the primary workflows for data preparation and model training:

- **`00_data_setup.ipynb`**: Generates the necessary synthetic datasets.
- **`01_training_baseline.ipynb`**: Handles the model baseline training and benchmarking process.

## Resources

- **Hugging Face Team**: [voyagersnlppolito](https://huggingface.co/voyagersnlppolito)
- **Weights & Biases Team**: [voyagers-polito](https://wandb.ai/voyagers-polito/huggingface)
