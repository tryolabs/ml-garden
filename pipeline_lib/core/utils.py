from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_error(y_true, y_pred, metric):
    metrics = {
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "root_mean_squared_error": lambda y_true, y_pred: mean_squared_error(
            y_true, y_pred, squared=False
        ),
    }

    if metric not in metrics:
        raise ValueError(f"Unsupported objective metric: {metric}")

    return metrics[metric](y_true, y_pred)
