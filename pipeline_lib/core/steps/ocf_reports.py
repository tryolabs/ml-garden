import matplotlib.colors
import matplotlib.pyplot as plt
import pandas as pd

from pipeline_lib.core import DataContainer
from pipeline_lib.core.steps.base import PipelineStep

# setting Tryo colors to use in plots

# primary Tryo colors
matplotlib.colors.ColorConverter.colors["tryo_eucalyptus"] = "#2CDDB3"
matplotlib.colors.ColorConverter.colors["tryo_ultramarine"] = "#4056F4"


class ocfReportsStep(PipelineStep):
    """Generate reports."""

    used_for_prediction = True
    used_for_training = False

    def __init__(self) -> None:
        """Initialize CalculateMetricsStep."""
        super().__init__()
        self.init_logger()

    def _calculate_abs_errors(self, y_true, y_pred):
        abs_errors = abs(y_true - y_pred)
        return abs_errors

    def _group_errors(self, df, target_column, criterion):
        average_error = df.groupby(criterion)["abs_error"].mean()
        average_value = df.groupby(criterion)[target_column].mean()
        grouped_errors = pd.merge(average_error, average_value, on=criterion)
        return grouped_errors

    def _plot_grouped_mae(self, df, target_column, criterion):
        # Plot for Average Generation and Average Error
        bar_width = 0.35
        index = range(df.shape[0])
        plt.figure(figsize=(30, 20))
        plt.rcParams.update({"font.size": 20})

        plt.bar(
            index,
            df[target_column],
            bar_width,
            label=f"{target_column}",
            color="tryo_eucalyptus",
        )
        plt.bar(
            [i + bar_width for i in index],
            df["abs_error"],
            bar_width,
            label=f"MAE by {criterion}",
            color="tryo_ultramarine",
        )

        plt.xlabel(f"{criterion}", fontsize=30)
        plt.title(f"MAE by {criterion}", fontsize=40)
        if len(index) > 500:
            plt.xticks([])
        else:
            plt.xticks([i + bar_width / 2 for i in index], df.index, rotation=45)
        plt.legend(fontsize=30)

        plt.show()

    def execute(self, data: DataContainer) -> DataContainer:
        self.logger.debug("Starting report generation")
        target_column_name = data.target
        if target_column_name is None:
            raise ValueError("Target column not found on any configuration.")
        df = data.flow
        df["predictions"] = data.predictions
        df["abs_error"] = self._calculate_abs_errors(df[target_column_name], df["predictions"])
        for criterion in ["date_month", "ss_id"]:
            mae_df = self._group_errors(df, target_column_name, criterion)
            self._plot_grouped_mae(mae_df, target_column_name, criterion)
        return data
