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
import pandas as pd  # Added pandas for CSV operations


# Define model with positional encoding (without multi-head attention)
class DecoderOnlyModel(nn.Module):
    def __init__(self, input_dim=15, target_dim=9, pos_emb_dim=32):
        super().__init__()
        self.input_dim = input_dim
        self.target_dim = target_dim
        self.pos_emb_dim = pos_emb_dim
        self.total_dim = input_dim + target_dim

        # Positional embedding layer
        self.pos_emb = nn.Embedding(self.total_dim, pos_emb_dim)

        # Feature projection layer (handles original features + positional embeddings)
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

        # Get positional embeddings [batch_size, seq_len, pos_emb_dim]
        pos_embeddings = self.pos_emb(positions)

        # Reshape input features to 3D and add positional embeddings
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


# Modified scatter plot visualization function - added PDF saving
def plot_scatter_comparison(y_true, y_pred, target_names):
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
    os.makedirs("plots", exist_ok=True)
    base_path = "/home/huangl/myprojects/mos_model_nn/picture"

    # PNG version
    scatter_png_path = os.path.join(base_path, "scatter_comparison.png")
    plt.savefig(scatter_png_path, bbox_inches='tight')
    print(f"Scatter plots (PNG) saved to: {scatter_png_path}")

    # PDF version
    scatter_pdf_path = os.path.join(base_path, "scatter_comparison.pdf")
    plt.savefig(scatter_pdf_path, bbox_inches='tight', format='pdf')
    print(f"Scatter plots (PDF) saved to: {scatter_pdf_path}")

    plt.show()
    plt.close()


# Modified VGS comparison plot function - added custom color support
def plot_vgs_comparison(X_test, y_true, y_pred, target_names, input_head):
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
        0.5: ('red', 'blue'),   # T_vds=0.5: red, pre_vds=0.5: blue
        1: ('green', 'black')   # T_vds=1: green, pre_vds=1: purple
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
        base_path = "/home/huangl/myprojects/mos_model_nn/picture/fine"

        # Create filename-safe target name
        safe_target = target.replace(' ', '_').replace('/', '_')

        # PDF version
        vgs_pdf_path = os.path.join(base_path, f"vgs_comparison_{safe_target}.pdf")
        plt.savefig(vgs_pdf_path, bbox_inches='tight', format='pdf')
        print(f"{target} VGS comparison plot (PDF) saved to: {vgs_pdf_path}")

        plt.close()


# Random seed setting
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Save predictions to CSV
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


# Main function
def main():
    set_seed()
    device_id = 2  # Specify GPU 1
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # ================== Configuration Parameters ==================
    batch_size = 65536
    finetune_epochs = 500

    # ================== Data Loading Configuration ==================
    # Load training, validation and test sets
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
    valid_files = ["/home/huangl/myprojects/mos_model_nn/data/Valid/finfet_7nm_2000/nch_svt_2000/output.csv"]
    test_files = ["/home/huangl/myprojects/mos_model_nn/data/Test/finfet_7nm_20000/nch_svt_vds=1/output.csv"]

    # ================== Data Loading ==================
    print("Loading training data for fitting scalers...")
    X_train, y_train_full, input_head, output_head = load_dataset(train_files)
    print("Loading validation data for fine-tuning...")
    X_valid, y_valid_full, _, _ = load_dataset(valid_files)
    print("Loading test data...")
    X_test, y_test_full, _, _ = load_dataset(test_files)

    TARGETS = ['qg', 'qb', 'qd', 'ids', 'didsdvg', 'didsdvd', 'cgg', 'cgd', 'cgs']
    LOG_TARGETS = ['ids', 'didsdvg', 'didsdvd']  # Targets requiring log transformation
    ABS_TARGETS = ['ids', 'didsdvg', 'didsdvd']  # Targets requiring absolute value

    # ================== Data Processing ==================
    # Take absolute value for specific targets
    print("\nApplying absolute value to specific targets...")
    abs_indices = [output_head.index(t) for t in ABS_TARGETS]
    for i in abs_indices:
        y_train_full[:, i] = np.abs(y_train_full[:, i])
        y_valid_full[:, i] = np.abs(y_valid_full[:, i])
        y_test_full[:, i] = np.abs(y_test_full[:, i])

    # Extract target columns
    y_train = y_train_full[:, [output_head.index(t) for t in TARGETS]]
    y_valid = y_valid_full[:, [output_head.index(t) for t in TARGETS]]
    y_test = y_test_full[:, [output_head.index(t) for t in TARGETS]]

    # Filter out samples with zero values
    print("\nFiltering samples where any target is zero...")
    non_zero_mask_train = np.all(y_train != 0, axis=1)
    non_zero_mask_valid = np.all(y_valid != 0, axis=1)
    non_zero_mask_test = np.all(y_test != 0, axis=1)

    X_train, y_train = X_train[non_zero_mask_train], y_train[non_zero_mask_train]
    X_valid, y_valid = X_valid[non_zero_mask_valid], y_valid[non_zero_mask_valid]
    X_test, y_test = X_test[non_zero_mask_test], y_test[non_zero_mask_test]

    print(f"Training samples after filtering: {X_train.shape[0]}")
    print(f"Fine-tuning samples after filtering: {X_valid.shape[0]}")
    print(f"Test samples after filtering: {X_test.shape[0]}")

    # Apply log10 transformation to specific targets
    print("\nApplying log10 transformation to specific targets...")
    log_indices = [TARGETS.index(t) for t in LOG_TARGETS]
    for i in log_indices:
        y_train[:, i] = np.log10(y_train[:, i])
        y_valid[:, i] = np.log10(y_valid[:, i])
        y_test[:, i] = np.log10(y_test[:, i])

    # ================== Data Standardization ==================
    # Fit scaler using training set
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)

    # Transform validation and test sets using training scaler
    X_valid_scaled = scaler_X.transform(X_valid)
    X_test_scaled = scaler_X.transform(X_test)

    # Create separate scalers for each target, fit on training set
    scalers_y = []
    y_train_scaled = np.zeros_like(y_train)
    y_valid_scaled = np.zeros_like(y_valid)
    y_test_scaled = np.zeros_like(y_test)

    for i in range(len(TARGETS)):
        scaler = StandardScaler()
        # Fit scaler only on training set
        y_train_scaled[:, i] = scaler.fit_transform(y_train[:, i].reshape(-1, 1)).flatten()
        # Transform validation and test sets using training scaler
        y_valid_scaled[:, i] = scaler.transform(y_valid[:, i].reshape(-1, 1)).flatten()
        y_test_scaled[:, i] = scaler.transform(y_test[:, i].reshape(-1, 1)).flatten()
        scalers_y.append(scaler)
        print(f"Target {TARGETS[i]} - Mean: {scaler.mean_[0]:.4f}, Scale: {scaler.scale_[0]:.4f}")

    # ================== Model Definition ==================
    model = DecoderOnlyModel(
        input_dim=X_train_scaled.shape[1],
        target_dim=len(TARGETS),
        pos_emb_dim=32
    ).to(device)

    # Model paths
    pretrain_model_path = "/home/huangl/myprojects/mos_model_nn/model/best_pretrain_model_all.pth"
    finetune_model_path = "/home/huangl/myprojects/mos_model_nn/model/best_finetune_model_n.pth"

    # ================== Load Pretrained Model ==================
    print(f"\nLoading pretrained model from: {pretrain_model_path}")
    model.load_state_dict(torch.load(pretrain_model_path))
    print("Pretrained model loaded successfully")

    criterion = nn.MSELoss()
    # ================== Fine-tuning Phase ==================
    print("\nStarting fine-tuning...")
    ft_optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_finetune_loss = float('inf')
    finetune_losses = []  # Record fine-tuning losses

    # Use validation set for fine-tuning
    finetune_data = X_valid_scaled
    finetune_targets = y_valid_scaled

    for epoch in range(finetune_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        indices = np.random.permutation(len(finetune_data))

        for batch_start in range(0, len(finetune_data), batch_size):
            batch_indices = indices[batch_start:batch_start + batch_size]
            X_batch = finetune_data[batch_indices]
            y_batch = finetune_targets[batch_indices]

            ft_optimizer.zero_grad()

            # Initialize input with zeros for targets
            inputs = np.hstack((X_batch, np.zeros((len(batch_indices), len(TARGETS)))))
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)

            # Create ground truth tensor
            true_tensor = torch.tensor(y_batch, dtype=torch.float32).to(device)

            # Generate independent random orders for each sample
            orders = np.array([np.random.permutation(len(TARGETS)) for _ in range(len(X_batch))])

            # Store predictions for each target - using tensor
            preds = torch.zeros_like(true_tensor)

            # Predict each target step by step
            for step in range(len(TARGETS)):
                # Get current target index for each sample in this step
                current_targets = orders[:, step]

                # Forward pass
                outputs = model(inputs)

                # Collect predictions for current step - keep tensor in computation graph
                row_indices = torch.arange(len(outputs)).to(device)
                col_indices = torch.tensor(current_targets).to(device)
                preds[row_indices, col_indices] = outputs[row_indices, col_indices]

                # Update input with ground truth values
                inputs[row_indices, X_batch.shape[1] + col_indices] = true_tensor[row_indices, col_indices]

            # Calculate loss
            loss = criterion(preds, true_tensor)

            loss_value = loss.item()
            epoch_loss += loss_value
            num_batches += 1

            loss.backward()
            ft_optimizer.step()

        # Calculate epoch average loss
        avg_epoch_loss = epoch_loss / num_batches
        finetune_losses.append(avg_epoch_loss)  # Record loss

        # Save best fine-tuned model
        if avg_epoch_loss < best_finetune_loss:
            best_finetune_loss = avg_epoch_loss
            torch.save(model.state_dict(), finetune_model_path)
            print(f"Saved new best fine-tuned model, loss: {best_finetune_loss:.6f}")

        if epoch % 10 == 0:
            print(f"Finetune Epoch {epoch} | Loss: {avg_epoch_loss:.4f}")

    # Load fine-tuned model
    print(f"\nLoading fine-tuned model")
    model.load_state_dict(torch.load(finetune_model_path))

    # ================== Testing Phase ==================
    model.eval()
    with torch.no_grad():
        pred_scaled = np.zeros_like(y_test_scaled)

        for batch_start in range(0, len(X_test_scaled), batch_size):
            batch_end = batch_start + batch_size
            X_batch = X_test_scaled[batch_start:batch_end]

            # Initialize input with zeros for targets
            test_inputs = np.hstack((X_batch, np.zeros((len(X_batch), len(TARGETS)))))
            test_inputs = torch.tensor(test_inputs, dtype=torch.float32).to(device)

            order = np.arange(len(TARGETS))  # Fixed order
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

        # Inverse scaling
        y_pred_log = np.zeros_like(pred_scaled)
        for i in range(len(TARGETS)):
            y_pred_log[:, i] = scalers_y[i].inverse_transform(
                pred_scaled[:, i].reshape(-1, 1)).flatten()

        # Inverse transform for log targets
        y_pred = y_pred_log.copy()
        for i in log_indices:
            y_pred[:, i] = 10 ** y_pred[:, i]

        # Restore original test set targets
        y_true = np.zeros_like(y_test)
        for i in range(len(TARGETS)):
            if i in log_indices:
                y_true[:, i] = 10 ** y_test[:, i]
            else:
                y_true[:, i] = y_test[:, i]

    # ================== Save Predictions to CSV ==================
    # Create path for prediction results
    predictions_path = "/home/huangl/myprojects/mos_model_nn/results/fine_csv"
    csv_filename = os.path.join(predictions_path, "model_predictions.csv")

    # Save predictions
    save_predictions_to_csv(X_test, y_pred, input_head, TARGETS, csv_filename)

    # ================== Visualization and Evaluation ==================
    print("\nGenerating scatter plots...")
    plot_scatter_comparison(y_true, y_pred, TARGETS)

    print("\nGenerating VGS comparison plots...")
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
