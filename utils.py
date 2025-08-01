import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from read_csv import read_csv


def load_dataset(file_list):
    all_inputs = []
    all_outputs = []
    input_head = None
    output_head = None

    for file in file_list:
        i_head, inputs, o_head, outputs = read_csv(file)
        if input_head is None:
            input_head = i_head
            output_head = o_head
        # Verify file header consistency
        assert i_head == input_head, "Input feature headers do not match"
        assert o_head == output_head, "Output feature headers do not match"

        all_inputs.extend(inputs)
        all_outputs.extend(outputs)

    return (np.array(all_inputs, dtype=np.float64),
            np.array(all_outputs, dtype=np.float64),
            input_head,
            output_head)


def calculate_metrics(y_true, y_pred, target_names):
    metrics = {}
    for i, target in enumerate(target_names):
        # Get column index by target name
        col_idx = target_names.index(target)

        # Special handling for qd targets: ignore samples with true value < 0.05e-16
        if target in ['qd', 'qg', 'qb']:
            # Filter using absolute values
            condition = np.abs(y_true[:, col_idx]) >= 0.05e-16
            y_t = y_true[condition, col_idx]
            y_p = y_pred[condition, col_idx]

            # Check if there are valid samples
            if len(y_t) == 0:
                metrics[target] = {
                    'MSE': np.nan,
                    'RMSE': np.nan,
                    'R2': np.nan,
                    'MRE(%)': np.nan
                }
                continue
        # Special handling for 'ids', 'didsdvg', 'didsdvd': ignore abs <= threshold
        elif target in ['ids', 'didsdvg']:
            # Filter using absolute values
            condition = np.abs(y_true[:, col_idx]) > 0.0002
            y_t = y_true[condition, col_idx]
            y_p = y_pred[condition, col_idx]

            # Check if there are valid samples
            if len(y_t) == 0:
                metrics[target] = {
                    'MSE': np.nan,
                    'RMSE': np.nan,
                    'R2': np.nan,
                    'MRE(%)': np.nan
                }
                continue
        elif target in ['didsdvd']:
            # Filter using absolute values
            condition = np.abs(y_true[:, col_idx]) > 0.00005
            y_t = y_true[condition, col_idx]
            y_p = y_pred[condition, col_idx]

            # Check if there are valid samples
            if len(y_t) == 0:
                metrics[target] = {
                    'MSE': np.nan,
                    'RMSE': np.nan,
                    'R2': np.nan,
                    'MRE(%)': np.nan
                }
                continue
        else:
            # For other targets, use non-zero samples
            non_zero = y_true[:, col_idx] != 0
            y_t = y_true[non_zero, col_idx]
            y_p = y_pred[non_zero, col_idx]

            if len(y_t) == 0:
                metrics[target] = {
                    'MSE': np.nan,
                    'RMSE': np.nan,
                    'R2': np.nan,
                    'MRE(%)': np.nan
                }
                continue

        # Calculate metrics
        mse = mean_squared_error(y_t, y_p)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_t, y_p)

        # MRE calculation (for qd with special filtering)
        mre = np.mean(np.abs((y_t - y_p) / y_t)) * 100

        metrics[target] = {
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MRE(%)': mre
        }
    return metrics


def plot_scatter_comparison(y_true, y_pred, target_names, save_path):
    plt.figure(figsize=(20, 15), dpi=100)
    for i, target in enumerate(target_names):
        plt.subplot(3, 3, i + 1)  # 3x3 grid layout

        # Get column index by target name
        col_idx = target_names.index(target)

        # Filter out zero values (safety check consistent with original)
        non_zero = y_true[:, col_idx] != 0
        y_t = y_true[non_zero, col_idx]
        y_p = y_pred[non_zero, col_idx]

        # Calculate statistical metrics
        r2 = r2_score(y_t, y_p)
        mse = mean_squared_error(y_t, y_p)

        # Plot scatter
        plt.scatter(y_t, y_p, alpha=0.5, label=f'R²={r2:.4f}\nMSE={mse:.2e}')
        max_val = max(y_t.max(), y_p.max())
        min_val = min(y_t.min(), y_p.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)

        # Set axis labels and tick font size
        plt.xlabel('True Value', fontsize=15)
        plt.ylabel('Predicted Value', fontsize=15)
        plt.title(target, fontsize=20)
        plt.legend(fontsize=20)
        plt.grid(True)

        # Increase tick label font size
        plt.tick_params(axis='both', which='major', labelsize=25)
        plt.tick_params(axis='both', which='minor', labelsize=25)

    plt.tight_layout()

    # Save plots - both PNG and PDF
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # PNG version
    scatter_png_path = os.path.join(save_path, "scatter_comparison.png")
    plt.savefig(scatter_png_path, bbox_inches='tight')
    print(f"Scatter plots (PNG) saved to: {scatter_png_path}")

    # PDF version
    scatter_pdf_path = os.path.join(save_path, "scatter_comparison.pdf")
    plt.savefig(scatter_pdf_path, bbox_inches='tight', format='pdf')
    print(f"Scatter plots (PDF) saved to: {scatter_pdf_path}")

    plt.show()
    plt.close()


def plot_vgs_comparison(X_test, y_true, y_pred, target_names, input_head, save_path):
    # Get indices for vgs and vds in input features
    vgs_idx = input_head.index('vgs') if 'vgs' in input_head else None
    vds_idx = input_head.index('vds') if 'vds' in input_head else None

    if vgs_idx is None or vds_idx is None:
        print("Warning: 'vgs' or 'vds' not found in input features, skipping VGS plots")
        return

    vgs_values = X_test[:, vgs_idx]
    vds_values = X_test[:, vds_idx]

    # Get unique vds values
    unique_vds = np.unique(vds_values)
    print(f"Found {len(unique_vds)} unique VDS values: {unique_vds}")

    # ====== Custom color scheme ======
    # Define custom colors for specific Vds values
    custom_colors = {
        # Format: Vds_value: (true_value_color, predicted_value_color)
        0.5: ('red', 'blue'),  # T_vds=0.5: red, pre_vds=0.5: blue
        1: ('green', 'black')  # T_vds=1: green, pre_vds=1: purple
    }

    # Create color map for other Vds values
    other_vds = [v for v in unique_vds if v not in custom_colors]
    if other_vds:
        norm = mcolors.Normalize(vmin=min(other_vds), vmax=max(other_vds))
        color_map = cm.viridis
    else:
        norm = None
        color_map = None

    # Create separate plots for each target and save
    for i, target in enumerate(target_names):
        # Create new figure
        plt.figure(figsize=(10, 8), dpi=100)  # Adjusted size to 10x8

        # Get column index for current target
        col_idx = target_names.index(target)

        # Plot curves for each vds value
        for vds in unique_vds:
            # Filter samples for current vds
            mask = vds_values == vds
            if np.sum(mask) == 0:
                continue

            vgs_current = vgs_values[mask]
            y_true_current = y_true[mask, col_idx]
            y_pred_current = y_pred[mask, col_idx]

            # Sort by vgs to ensure continuous curves
            sort_idx = np.argsort(vgs_current)
            vgs_sorted = vgs_current[sort_idx]
            y_true_sorted = y_true_current[sort_idx]
            y_pred_sorted = y_pred_current[sort_idx]

            # Get colors for current vds
            if vds in custom_colors:
                true_color, pred_color = custom_colors[vds]
            else:
                if norm and color_map:
                    color_val = norm(vds)
                    true_color = color_map(color_val)
                    pred_color = color_map(color_val)
                else:
                    true_color = 'blue'
                    pred_color = 'red'

            # Plot true and predicted values vs Vgs
            plt.plot(vgs_sorted, y_true_sorted, 'o-', markersize=4,
                     label=f'T_vds={vds:.2f}', color=true_color, alpha=0.7)
            plt.plot(vgs_sorted, y_pred_sorted, 'x-', markersize=4,
                     label=f'pre_vds={vds:.2f}', color=pred_color, alpha=0.7, linestyle='--')

        # Set axis labels and tick font size
        plt.xlabel('Vgs', fontsize=20)
        plt.ylabel(target, fontsize=20)
        plt.title(f'{target} vs Vgs', fontsize=20)
        plt.grid(True)

        # Increase tick label font size
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.tick_params(axis='both', which='minor', labelsize=20)

        # Add legend
        handles, labels = plt.gca().get_legend_handles_labels()
        # Simplify legend: show only one entry per vds value
        unique_labels = {}
        for handle, label in zip(handles, labels):
            # Remove duplicate labels
            if label not in unique_labels:
                unique_labels[label] = handle

        # Create custom legend
        plt.legend(unique_labels.values(), unique_labels.keys(), fontsize=14, loc='best',
                   ncol=2)  # Display in two columns

        plt.tight_layout()

        # Save plot - separate PDF for each target
        os.makedirs(save_path, exist_ok=True)

        # Create filename-safe target name
        safe_target = target.replace(' ', '_').replace('/', '_')

        # PDF version
        vgs_pdf_path = os.path.join(save_path, f"vgs_comparison_{safe_target}.pdf")
        plt.savefig(vgs_pdf_path, bbox_inches='tight', format='pdf')
        print(f"{target} VGS comparison plot (PDF) saved to: {vgs_pdf_path}")

        plt.close()


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_predictions_to_csv(X_test, y_pred, input_head, target_names, filename):
    """
    Save input features and predictions to CSV file

    Parameters:
        X_test: Test set input features (numpy array)
        y_pred: Predictions (numpy array)
        input_head: Input feature names list
        target_names: Target names list
        filename: CSV file path to save
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Create DataFrame for input features
    df_inputs = pd.DataFrame(X_test, columns=input_head)

    # Create DataFrame for predictions
    df_predictions = pd.DataFrame(y_pred, columns=target_names)

    # Combine both DataFrames
    df_combined = pd.concat([df_inputs, df_predictions], axis=1)

    # Save to CSV
    df_combined.to_csv(filename, index=False)
    print(f"Predictions saved to: {filename}")