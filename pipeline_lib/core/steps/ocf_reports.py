import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_error

from pipeline_lib.core import DataContainer
from pipeline_lib.core.model import Model
from pipeline_lib.core.steps.base import PipelineStep

# setting Tryo colors to use in plots

# primary Tryo colors
matplotlib.colors.ColorConverter.colors["tryo_eucalyptus"] = "#2CDDB3"
matplotlib.colors.ColorConverter.colors["tryo_ultramarine"] = "#4056F4"

# secondary Tryo colors
matplotlib.colors.ColorConverter.colors["tryo_navy_blue"] = "#1709FF"
matplotlib.colors.ColorConverter.colors["tryo_blue_violet"] = "#6207F3"
matplotlib.colors.ColorConverter.colors["tryo_carmine_pink"] = "#FF3139"

# neutral Tryo colors (gray, black)
matplotlib.colors.ColorConverter.colors["tryo_bright_gray"] = "#E5EAEF"
matplotlib.colors.ColorConverter.colors["tryo_french_middle"] = "#0B0723"


class ocfReportsStep(PipelineStep):
    """Generate reports."""

    used_for_prediction = True
    used_for_training = False

    def __init__(self) -> None:
        """Initialize CalculateMetricsStep."""
        super().__init__()
        self.init_logger()

    def _grouped_mae(self, df):
        grouped_mae_results = []

        for month in df["date_month"].unique():
            df_month = df[df["date_month"] == month]
            y_true = df_month["average_power_kw"]
            y_pred = df_month["predictions"]
            av_gen = y_true.mean()
            mae = mean_absolute_error(y_true, y_pred)
            grouped_mae_results.append((month, av_gen, mae))

        mae_df = pd.DataFrame(grouped_mae_results, columns=["date_month", "av_generation", "MAE"])

        return mae_df

    def _plot_grouped_mae(self, df):
        # Plot for Average Generation and Average Error
        bar_width = 0.35
        index = range(df.shape[0])
        plt.figure(figsize=(30, 20))
        plt.rcParams.update({"font.size": 20})

        bar1 = plt.bar(
            index,
            df["av_generation"],
            bar_width,
            label="Average generation (kW)",
            color="tryo_eucalyptus",
        )
        bar2 = plt.bar(
            [i + bar_width for i in index],
            df["MAE"],
            bar_width,
            label="MAE (kW)",
            color="tryo_ultramarine",
        )

        plt.xlabel("Month", fontsize=30)
        plt.ylabel("Generation/MAE (kW)", fontsize=30)
        plt.title("MAE by month", fontsize=40)

        plt.xticks([i + bar_width / 2 for i in index], df["date_month"], rotation=45)
        plt.legend(fontsize=30)

        plt.show()

    def execute(self, data: DataContainer) -> DataContainer:
        self.logger.debug("Starting report generation")
        target_column_name = data.target
        if target_column_name is None:
            raise ValueError("Target column not found on any configuration.")

        df = data.flow[["date_month", target_column_name]]
        df["predictions"] = data.predictions
        mae_df = self._grouped_mae(df)
        self._plot_grouped_mae(mae_df)
        return data
