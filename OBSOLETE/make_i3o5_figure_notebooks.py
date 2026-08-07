import json
import shutil
from pathlib import Path


ROOT = Path(r"d:\Research\NO-2D-Metamaterials")


CELL4_NEW = """def visualize_sample(input_tensor, output_tensor, target_tensor):
    \"""
    Visualize input and output tensors from a single sample.

    Args:
        input_tensor: Tensor of shape (3, H, W) containing input components
        output_tensor: Tensor of shape (C, H, W) containing output components
        (optional) target_tensor: Tensor of shape (C, H, W) containing target components
    \"""
    # Create figure for input components
    fig1 = plt.figure(figsize=(12, 4))

    # Plot input tensor components (1x3 subplot)
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        # Convert to float32 if necessary to avoid unsupported dtype errors
        tensor_data = input_tensor[i].abs()
        if tensor_data.dtype in [torch.float8_e4m3fn, torch.float16]:
            tensor_data = tensor_data.float()
        im = plt.imshow(tensor_data.numpy())
        plt.colorbar(im)
        plt.title(f"Input Component {i + 1}")

    plt.tight_layout()
    plt.show()

    # Create figure for output components
    num_outputs = output_tensor.shape[0]
    fig2 = plt.figure(figsize=(4 * num_outputs, 4))

    # Plot output tensor components (1xC subplot)
    for i in range(num_outputs):
        plt.subplot(1, num_outputs, i + 1)
        # Convert to float32 if necessary to avoid unsupported dtype errors
        tensor_data = output_tensor[i].abs()
        if tensor_data.dtype in [torch.float8_e4m3fn, torch.float16]:
            tensor_data = tensor_data.float()
        im = plt.imshow(tensor_data.numpy())
        plt.colorbar(im)
        plt.title(f"Output Component {i + 1}")

    plt.tight_layout()
    plt.show()

    # Create figure for target components
    if target_tensor is not None:
        num_targets = target_tensor.shape[0]
        fig3 = plt.figure(figsize=(4 * num_targets, 4))
        for i in range(num_targets):
            plt.subplot(1, num_targets, i + 1)
            # Convert to float32 if necessary to avoid unsupported dtype errors
            tensor_data = target_tensor[i].abs()
            if tensor_data.dtype in [torch.float8_e4m3fn, torch.float16]:
                tensor_data = tensor_data.float()
            im = plt.imshow(tensor_data.numpy())
            plt.colorbar(im)
            plt.title(f"Target Component {i + 1}")

        plt.tight_layout()
        plt.show()
"""


CELL10_NEW = """for i, sample_idx in enumerate(random_indices):
    print(f"Visualizing sample {i} (index {sample_idx})")

    # Get the stored data from the earlier processing
    input_sample = all_inputs[i]  # [3, 32, 32]
    output_sample = all_predictions[i]  # [C, 32, 32]
    target_sample = all_targets[i]  # [C, 32, 32]
    num_channels = output_sample.shape[0]

    # Convert to numpy for plotting
    input_sample = input_sample.numpy()
    output_sample = output_sample.numpy()
    target_sample = target_sample.numpy()

    # Plot the 3 inputs in a 1x3 subplot
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(f"Sample {i} - Inputs", fontsize=20)

    input_titles = ["Unit Cell \\nGeometry", "(2D Embedded) \\n Wavevectors", "(1D Embedded) \\n Band"]

    for j in range(3):
        im = axes[j].imshow(input_sample[j], cmap="viridis")
        axes[j].set_title(input_titles[j], fontsize=18)
        axes[j].axis("off")
        cbar = plt.colorbar(im, ax=axes[j])
        cbar.ax.tick_params(labelsize=16)

    plt.tight_layout()
    plt.show()

    # Plot outputs, targets, and differences in a 3xC subplot with shared colorbar
    fig, axes = plt.subplots(3, num_channels, figsize=(4 * num_channels + 4, 12))
    fig.suptitle(f"Sample {i} - Outputs, Targets, and Differences", fontsize=20)

    # Channel labels
    default_labels = [
        "Displacement x_real",
        "Displacement x_imag",
        "Displacement y_real",
        "Displacement y_imag",
        "Eigenfrequency wavelet",
    ]
    channel_labels = default_labels[:num_channels]
    if num_channels > len(default_labels):
        channel_labels.extend([f"Channel {k + 1}" for k in range(len(default_labels), num_channels)])

    # Find global min/max for outputs and targets to share colorbar
    output_min, output_max = output_sample.min(), output_sample.max()
    target_min, target_max = target_sample.min(), target_sample.max()

    # Include differences in the global min/max calculation
    diff_data = []
    for j in range(num_channels):
        diff = output_sample[j] - target_sample[j]
        diff_data.append(diff)
    diff_min = min([d.min() for d in diff_data])
    diff_max = max([d.max() for d in diff_data])

    # Global min/max across all data (outputs, targets, and differences)
    global_min = min(output_min, target_min, diff_min)
    global_max = max(output_max, target_max, diff_max)

    # Row 1: Outputs
    for j in range(num_channels):
        im = axes[0, j].imshow(output_sample[j], cmap="viridis", vmin=global_min, vmax=global_max)
        axes[0, j].set_title(f"Output \\n{channel_labels[j]}", fontsize=18)
        axes[0, j].axis("off")

    # Row 2: Targets
    for j in range(num_channels):
        im = axes[1, j].imshow(target_sample[j], cmap="viridis", vmin=global_min, vmax=global_max)
        axes[1, j].set_title(f"Target \\n{channel_labels[j]}", fontsize=18)
        axes[1, j].axis("off")

    # Row 3: Differences (Output - Target)
    for j in range(num_channels):
        diff = output_sample[j] - target_sample[j]
        im = axes[2, j].imshow(diff, cmap="viridis", vmin=global_min, vmax=global_max)
        axes[2, j].set_title(f"Difference \\n{channel_labels[j]}", fontsize=18)
        axes[2, j].axis("off")

    # Adjust layout to make room for colorbar
    plt.subplots_adjust(right=0.85, hspace=0.3)

    # Add single shared colorbar on the right side
    cbar_ax = fig.add_axes([0.87, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=16)

    plt.show()
"""


CELL6_OLD_NEW = """# Create arrays to store all pixel values across dataset
input_pixels = [[] for _ in range(3)]  # 3 input components
num_output_components = dataset[0][1].shape[0]
output_pixels = [[] for _ in range(num_output_components)]

# Sample a subset of the dataset for efficiency
num_samples = min(1000, len(dataset))
sample_indices = np.random.choice(len(dataset), num_samples, replace=False)

# Collect pixel values
for idx in sample_indices:
    sample = dataset[idx]
    input_tensor = sample[0]
    output_tensor = sample[1]

    # Gather input pixels
    for i in range(3):
        input_pixels[i].extend(input_tensor[i].numpy().flatten())

    # Gather output pixels
    for i in range(num_output_components):
        output_pixels[i].extend(output_tensor[i].numpy().flatten())

# Plot input histograms
fig1 = plt.figure(figsize=(12, 4))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.hist(input_pixels[i], bins=100)
    plt.title(f"Input Component {i + 1} Distribution")
    plt.xlabel("Pixel Value")
    plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Plot output histograms
fig2 = plt.figure(figsize=(4 * num_output_components, 4))
for i in range(num_output_components):
    plt.subplot(1, num_output_components, i + 1)
    plt.hist(output_pixels[i], bins=100)
    plt.title(f"Output Component {i + 1} Distribution")
    plt.xlabel("Pixel Value")
    plt.ylabel("Count")
plt.tight_layout()
plt.show()
"""


CELL12_OLD_NEW = """def plot_sample(data, title, labels, shared_colorbar=False, vmin=None, vmax=None, layout=None, save=False, save_path='plot.png'):
    # Automatically select layout based on the number of data arrays if not provided
    num_arrays = data.shape[0]
    if layout is None:
        if num_arrays == 3:
            layout = (1, 3)
        elif num_arrays == 4:
            layout = (2, 2)
        elif num_arrays == 5:
            layout = (1, 5)
        else:
            raise ValueError('Unsupported number of arrays for plotting.')

    # Adjust figure size based on layout
    if layout == (1, 3):
        fig, axes = plt.subplots(*layout, figsize=(12, 4))
    elif layout == (2, 2):
        fig, axes = plt.subplots(*layout, figsize=(8, 8))
    elif layout == (1, 5):
        fig, axes = plt.subplots(*layout, figsize=(20, 4))
    else:
        raise ValueError('Unsupported layout')

    for ax, (idx, label) in zip(axes.flatten(), labels):
        im = ax.imshow(data[idx, :, :].cpu(), cmap='viridis', vmin=vmin if shared_colorbar else None, vmax=vmax if shared_colorbar else None)
        ax.set_title(label)
        if not shared_colorbar:
            fig.colorbar(im, ax=ax)

    if shared_colorbar:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist())
        cbar.ax.set_ylabel('Color scale')

    plt.suptitle(title)

    if save:
        plt.savefig(save_path)
        print(f'Plot saved at {save_path}')
        plt.close(fig)
    else:
        plt.show()

def plot_inputs(data, title, labels, save=False, save_path='example_plot.png'):
    plot_sample(data, title, labels, shared_colorbar=True, layout=(1, 3), save=save, save_path=save_path)

def plot_predictions_and_targets(inputs, outputs, targets, save=False, save_dir='figures/', file_suffix='example'):
    if save:
        os.makedirs(save_dir, exist_ok=True)

    input_filename = f'{save_dir}/input_{file_suffix}.png'
    output_filename = f'{save_dir}/output_{file_suffix}.png'
    target_filename = f'{save_dir}/target_{file_suffix}.png'

    input_labels = [(0, 'geometry'), (1, 'waveform'), (2, 'band')]
    plot_inputs(inputs, 'Inputs', input_labels, save=save, save_path=input_filename)

    # Compute global min and max for shared colorbar between outputs and targets
    vmin = min(np.min(outputs.cpu().numpy()), np.min(targets.cpu().numpy()))
    vmax = max(np.max(outputs.cpu().numpy()), np.max(targets.cpu().numpy()))

    n_ch = outputs.shape[0]
    default_labels = ['eigenvector_x_real', 'eigenvector_x_imag', 'eigenvector_y_real', 'eigenvector_y_imag', 'eigenfrequency_wavelet']
    names = default_labels[:n_ch]
    if n_ch > len(default_labels):
        names.extend([f'channel_{i + 1}' for i in range(len(default_labels), n_ch)])

    prediction_labels = [(i, names[i]) for i in range(n_ch)]
    plot_sample(outputs, 'Model Predictions', prediction_labels, shared_colorbar=True, vmin=vmin, vmax=vmax, save=save, save_path=output_filename)

    target_labels = [(i, names[i]) for i in range(n_ch)]
    plot_sample(targets, 'Target Values', target_labels, shared_colorbar=True, vmin=vmin, vmax=vmax, save=save, save_path=target_filename)
"""


def lines(src: str) -> list[str]:
    return [line + "\n" for line in src.split("\n")[:-1]]


def save_notebook(path: Path, nb: dict) -> None:
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


def update_main_figure_notebook(src_name: str) -> Path:
    src = ROOT / src_name
    dst = ROOT / f"{src.stem}_I3O5.ipynb"
    shutil.copy2(src, dst)
    nb = json.loads(dst.read_text(encoding="utf-8"))

    nb["cells"][4]["source"] = lines(CELL4_NEW)

    cell8 = "".join(nb["cells"][8].get("source", []))
    cell8 = cell8.replace("out_channels=4", "out_channels=5")
    nb["cells"][8]["source"] = lines(cell8 + "\n")

    nb["cells"][10]["source"] = lines(CELL10_NEW)

    save_notebook(dst, nb)
    return dst


def update_old_datastructures_notebook() -> Path:
    src = ROOT / "figures_old_datastructures.ipynb"
    dst = ROOT / "figures_old_datastructures_I3O5.ipynb"
    shutil.copy2(src, dst)
    nb = json.loads(dst.read_text(encoding="utf-8"))

    nb["cells"][4]["source"] = lines(CELL4_NEW)
    nb["cells"][6]["source"] = lines(CELL6_OLD_NEW)
    nb["cells"][12]["source"] = lines(CELL12_OLD_NEW)

    for idx in (13, 22):
        s = "".join(nb["cells"][idx].get("source", []))
        s = s.replace("out_channels=4", "out_channels=5")
        nb["cells"][idx]["source"] = lines(s + "\n")

    save_notebook(dst, nb)
    return dst


def main() -> None:
    created = []
    for name in ("figures.ipynb", "figures_binary.ipynb", "figures_251117.ipynb"):
        created.append(update_main_figure_notebook(name))
    created.append(update_old_datastructures_notebook())

    print("Created notebooks:")
    for p in created:
        print(f" - {p}")


if __name__ == "__main__":
    main()
