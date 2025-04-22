from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
import numpy as np
import matplotlib.pyplot as plt

def plot_regressor_scores(y_preds, y_actuals, set_name=None):
    """Print the RMSE and MAE for the provided data

    Parameters
    ----------
    y_preds : Numpy Array
        Predicted target
    y_actuals : Numpy Array
        Actual target
    set_name : str
        Name of the set to be printed

    Returns
    -------
    """

    # Calculate RMSE and MAE
    rmse_value = np.sqrt(mse(y_actuals, y_preds))
    mae_value = mae(y_actuals, y_preds)


    # Print results
    print(f"RMSE {set_name}: {rmse_value:.2f}")
    print(f"MAE {set_name}: {mae_value:.2f}")


    # Plot expected vs actual
    plt.figure(figsize=(12, 6))
    plt.plot(y_actuals, label='Actual', color='blue', linewidth=2)
    plt.plot(y_preds, label='Predicted', color='orange', linewidth=2)
    plt.title(f'Actual vs Predicted ({set_name})')
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()