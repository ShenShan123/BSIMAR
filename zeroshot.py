import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from read_csv import read_csv
import os


# Define model with positional embedding (no multi-head attention)
class DecoderOnlyModel(nn.Module):
    def __init__(self, input_dim=15, target_dim=9, pos_emb_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.pos_emb_dim = pos_emb_dim
        self.total_dim = input_dim + target_dim

        # Position embedding layer
        self.pos_emb = nn.Embedding(self.total_dim, pos_emb_dim)

        # Feature projection layer (handles raw features + position embedding)
        self.feature_proj = nn.Linear(1 + pos_emb_dim, 1)

        # Decoder (remains unchanged)
        self.decoder = nn.Sequential(
            nn.Linear(self.total_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, target_dim)
        )

    def forward(self, x):
        batch_size, seq_len = x.shape

        # Create position indices [0, 1, 2, ..., total_dim-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)

        # Get position embeddings [batch_size, seq_len, pos_emb_dim]
        pos_embeddings = self.pos_emb(positions)

        # Reshape input features to 3D and add position embeddings
        x_reshaped = x.view(batch_size, seq_len, 1)  # [batch, seq, 1]
        x_with_pos = torch.cat([x_reshaped, pos_embeddings], dim=-1)  # [batch, seq, 1+pos_emb_dim]

        # Feature projection
        projected = self.feature_proj(x_with_pos)  # [batch, seq, 1]

        # Flatten the projected result after removing multi-head attention
        projected_flat = projected.squeeze(-1)  # [batch, seq]

        # Decoder
        return self.decoder(projected_flat)


# New data loading function
def load_dataset(file_list):
    all_inputs = []
    all_outputs = []
    input_head = None
    output_head = None

    for file in file_list:
        # Assume read_csv function is defined
        i_head, inputs, o_head, outputs = read_csv(file)
        if input_head is None:
            input_head = i_head
            output_head = o_head
        # Verify header consistency
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

        # Special handling for qd targets: ignore samples where true value absolute value < 0.05e-16
        if target in ['qd', 'qg', 'qb']:
            # Filter using absolute value
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
        # Special handling for 'ids', 'didsdvg', 'didsdvd' targets: ignore samples with absolute value <= 0.0002
        elif target in ['ids', 'didsdvg']:
            # Filter using absolute value
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
            # Filter using absolute value
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
            # For other targets use non-zero samples
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

        # MRE calculation (uses specially filtered samples for qd)
        mre = np.mean(np.abs((y_t - y_p) / y_t)) * 100

        metrics[target] = {
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2,
            'MRE(%)': mre
        }
    return metrics


# Modified scatter plot visualization function - add PDF saving
def plot_scatter_comparison(y_true, y_pred, target_names):
    plt.figure(figsize=(20, 15), dpi=100)
    for i, target in enumerate(target_names):
        plt.subplot(3, 3, i + 1)  # 3x3 grid layout

        # Get column index by target name
        col_idx = target_names.index(target)

        # Use boolean indexing to filter zero values (safe check consistent with original code)
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

        # Set axis labels and tick font sizes
        plt.xlabel('True Value', fontsize=15)
        plt.ylabel('Predicted Value', fontsize=15)
        plt.title(target, fontsize=20)
        plt.legend(fontsize=20)
        plt.grid(True)

        # Increase tick label font size
        plt.tick_params(axis='both', which='major', labelsize=25)
        plt.tick_params(axis='both', which='minor', labelsize=25)

    plt.tight_layout()

    # Save plots - both as PNG and PDF
    os.makedirs("plots", exist_ok=True)
    base_path = "/home/huangl/myprojects/mos_model_nn/picture"

    # PNG version
    scatter_png_path = os.path.join(base_path, "scatter_comparison.png")
    plt.savefig(scatter_png_path, bbox_inches='tight')
    print(f"Scatter plot (PNG) saved to: {scatter_png_path}")

    # PDF version
    scatter_pdf_path = os.path.join(base_path, "scatter_comparison.pdf")
    plt.savefig(scatter_pdf_path, bbox_inches='tight', format='pdf')
    print(f"Scatter plot (PDF) saved to: {scatter_pdf_path}")

    plt.show()
    plt.close()


# Modified VGS variation plot function - add custom color support
def plot_vgs_comparison(X_test, y_true, y_pred, target_names, input_head):
    # Get indices of vgs and vds in input features
    vgs_idx = input_head.index('vgs') if 'vgs' in input_head else None
    vds_idx = input_head.index('vds') if 'vds' in input_head else None

    if vgs_idx is None or vds_idx is None:
        print("Warning: 'vgs' or 'vds' not found in input features, skipping VGS plots")
        return

    plt.figure(figsize=(20, 15), dpi=100)
    vgs_values = X_test[:, vgs_idx]
    vds_values = X_test[:, vds_idx]

    # Get unique vds values
    unique_vds = np.unique(vds_values)
    print(f"Found {len(unique_vds)} unique VDS values: {unique_vds}")

    # ====== Custom color scheme ======
    # Define custom colors for specific Vds values
    custom_colors = {
        # Format: Vds value: (true color, prediction color)
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

    for i, target in enumerate(target_names):
        plt.subplot(3, 3, i + 1)  # 3x3 grid layout

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

            # Sort by vgs to ensure continuous curve
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

        # Set axis labels and tick font sizes
        plt.xlabel('Vgs', fontsize=20)
        plt.ylabel(target, fontsize=20)
        plt.title(f'{target} vs Vgs', fontsize=20)
        plt.grid(True)

        # Increase tick label font size
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.tick_params(axis='both', which='minor', labelsize=20)

        # Add legend (only once to avoid duplication)
        if i == 0:
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

    # Save plots - both as PNG and PDF
    base_path = "/home/huangl/myprojects/mos_model_nn/picture"

    # PNG version
    vgs_png_path = os.path.join(base_path, "vgs_comparison.png")
    plt.savefig(vgs_png_path, bbox_inches='tight')
    print(f"VGS variation plot (PNG) saved to: {vgs_png_path}")

    # PDF version
    vgs_pdf_path = os.path.join(base_path, "vgs_comparison.pdf")
    plt.savefig(vgs_pdf_path, bbox_inches='tight', format='pdf')
    print(f"VGS variation plot (PDF) saved to: {vgs_pdf_path}")

    plt.show()
    plt.close()


# Set random seed
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Main function - modified for test-only mode
def main():
    set_seed()
    device_id = 1  # Specify to use GPU 1
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # ================== Configuration Parameters ==================
    batch_size = 65536

    # ================== Data Loading Configuration ==================

    # Note: To preprocess correctly, we still need training data statistics
    # Therefore need to load training data to fit the scalers
    train_files = [
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs300000/nmos/SIM/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs300000/nmos/SIM1/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs300000/nmos/SIM2/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs300000/pmos/SIM3/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs300000/pmos/SIM4/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_7nm/Finfet_lhs_lhs300000/pmos/SIM5/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_12nm/Finfet_lhs_lhs300000/nmos/SIM/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_12nm/Finfet_lhs_lhs300000/nmos/SIM1/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_12nm/Finfet_lhs_lhs300000/nmos/SIM2/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_12nm/Finfet_lhs_lhs300000/pmos/SIM3/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_12nm/Finfet_lhs_lhs300000/pmos/SIM5/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs300000/nmos/SIM/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs300000/nmos/SIM1/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs300000/nmos/SIM2/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs300000/pmos/SIM3/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs300000/pmos/SIM4/output.csv",
        "/home/huangl/myprojects/mos_model_nn/data/FinFet_16nm/Finfet_lhs_lhs300000/pmos/SIM5/output.csv",
    ]
    test_files = ["/home/huangl/myprojects/mos_model_nn/data/Test/finfet_7nm_20000/nch_svt_2vds/output.csv"]
    # ================== Data Loading ==================
    print("Loading training data for fitting scalers...")
    X_train, y_train_full, input_head, output_head = load_dataset(train_files)
    print("Loading test data...")
    X_test, y_test_full, _, _ = load_dataset(test_files)

    TARGETS = ['qg', 'qb', 'qd', 'ids', 'didsdvg', 'didsdvd', 'cgg', 'cgd', 'cgs']
    LOG_TARGETS = ['ids', 'didsdvg', 'didsdvd']  # Targets requiring Log transformation
    ABS_TARGETS = ['ids', 'didsdvg', 'didsdvd']  # Targets requiring absolute value

    # ================== Data Processing ==================
    # Take absolute value for specific targets
    print("\nApplying absolute value to specific targets...")
    abs_indices = [output_head.index(t) for t in ABS_TARGETS]
    for i in abs_indices:
        y_train_full[:, i] = np.abs(y_train_full[:, i])
        y_test_full[:, i] = np.abs(y_test_full[:, i])

    # Extract target columns
    y_train = y_train_full[:, [output_head.index(t) for t in TARGETS]]
    y_test = y_test_full[:, [output_head.index(t) for t in TARGETS]]

    # Filter samples containing zero values
    print("\nFiltering samples where any target is zero...")
    non_zero_mask_train = np.all(y_train != 0, axis=1)
    non_zero_mask_test = np.all(y_test != 0, axis=1)

    X_train, y_train = X_train[non_zero_mask_train], y_train[non_zero_mask_train]
    X_test, y_test = X_test[non_zero_mask_test], y_test[non_zero_mask_test]

    print(f"Train samples after filtering: {X_train.shape[0]}")
    print(f"Test samples after filtering: {X_test.shape[0]}")

    # Apply Log10 transformation to specific targets
    print("\nApplying log10 transformation to specific targets...")
    log_indices = [TARGETS.index(t) for t in LOG_TARGETS]
    for i in log_indices:
        y_train[:, i] = np.log10(y_train[:, i])
        y_test[:, i] = np.log10(y_test[:, i])

    # ================== Data Standardization ==================
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)  # Fit only on training data
    X_test_scaled = scaler_X.transform(X_test)

    # Create separate scalers for each target
    scalers_y = []
    y_train_scaled = np.zeros_like(y_train)
    y_test_scaled = np.zeros_like(y_test)

    for i in range(len(TARGETS)):
        scaler = StandardScaler()
        y_train_scaled[:, i] = scaler.fit_transform(y_train[:, i].reshape(-1, 1)).flatten()
        y_test_scaled[:, i] = scaler.transform(y_test[:, i].reshape(-1, 1)).flatten()
        scalers_y.append(scaler)
        print(f"Target {TARGETS[i]} - Mean: {scaler.mean_[0]:.4f}, Scale: {scaler.scale_[0]:.4f}")

    # ================== Model Definition ==================
    model = DecoderOnlyModel(
        input_dim=X_train_scaled.shape[1],
        target_dim=len(TARGETS),
        pos_emb_dim=32
    ).to(device)

    # Load pretrained model
    pretrain_model_path = "/home/huangl/myprojects/mos_model_nn/model/dataL/best_pretrain_model.pth"
    print(f"\nLoading pretrained model: {pretrain_model_path}")
    state_dict = torch.load(pretrain_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode

    # ================== Testing Phase ==================
    print("\nStarting testing phase...")
    with torch.no_grad():
        pred_scaled = np.zeros_like(y_test_scaled)
        fixed_order = np.arange(len(TARGETS))  # Fixed target order [0,1,2,...,8]

        for batch_start in range(0, len(X_test_scaled), batch_size):
            batch_end = batch_start + batch_size
            X_batch = X_test_scaled[batch_start:batch_end]

            # Initialize input with all zeros for targets
            test_inputs = np.hstack((X_batch, np.zeros((len(X_batch), len(TARGETS)))))
            test_inputs = torch.tensor(test_inputs, dtype=torch.float32).to(device)

            # Use fixed order
            order = fixed_order
            preds = {}

            for step in order:
                outputs = model(test_inputs)
                pred = outputs[:, step]
                preds[step] = pred

                # Update input with current prediction
                test_inputs = test_inputs.clone()
                test_inputs[:, X_batch.shape[1] + step] = pred

            # Save results
            for t in range(len(TARGETS)):
                pred_t = preds[t].cpu().numpy()
                pred_scaled[batch_start:batch_end, t] = pred_t

        # Inverse standardization
        y_pred_log = np.zeros_like(pred_scaled)
        for i in range(len(TARGETS)):
            y_pred_log[:, i] = scalers_y[i].inverse_transform(
                pred_scaled[:, i].reshape(-1, 1)).flatten()

        # Inverse transform for Log targets
        y_pred = y_pred_log.copy()
        for i in log_indices:
            y_pred[:, i] = 10 ** y_pred[:, i]

        # Restore original test set target values
        y_true = np.zeros_like(y_test)
        for i in range(len(TARGETS)):
            if i in log_indices:
                y_true[:, i] = 10 ** y_test[:, i]
            else:
                y_true[:, i] = y_test[:, i]

    # ================== Result Visualization and Evaluation ==================
    print("\nGenerating scatter plots...")
    plot_scatter_comparison(y_true, y_pred, TARGETS)

    print("\nGenerating VGS variation plots...")
    plot_vgs_comparison(X_test, y_true, y_pred, TARGETS, input_head)

    print("\nGlobal evaluation metrics:")
    global_metrics = calculate_metrics(y_true, y_pred, TARGETS)

    # Calculate average metrics across all targets
    avg_metrics = {
        'MSE': 0.0,
        'RMSE': 0.0,
        'R2': 0.0,
        'MRE(%)': 0.0
    }
    valid_targets = 0

    for target, metrics in global_metrics.items():
        print(f"\n{target}:")
        for k, v in metrics.items():
            # Skip NaN values
            if not np.isnan(v):
                avg_metrics[k] += v
                # Count only valid targets
                if k == 'MSE':
                    valid_targets += 1
            print(f"{k}: {v:.4e}" if k != 'MRE(%)' else f"{k}: {v:.2f}%")

    # Calculate averages
    if valid_targets > 0:
        avg_metrics['MSE'] /= valid_targets
        avg_metrics['RMSE'] /= valid_targets
        avg_metrics['R2'] /= valid_targets
        avg_metrics['MRE(%)'] /= valid_targets

    # Print average metrics
    print("\nAverage metrics across all targets:")
    for metric, value in avg_metrics.items():
        if metric != 'MRE(%)':
            print(f"Average {metric}: {value:.4e}")
        else:
            print(f"Average {metric}: {value:.2f}%")


if __name__ == "__main__":
    main()
