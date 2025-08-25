from io import BytesIO
import urllib.request
from zipfile import ZipFile
import os
import torch
import torch.utils.data
from torchvision import datasets, transforms
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import cv2
import numpy as np
import torch.nn as nn
import random





# Let's see if we have an available GPU


def setup_env():
    use_cuda = torch.cuda.is_available()

    if use_cuda:
        print("GPU available")
    else:
        print("GPU *NOT* available. Will use CPU (slow)")

    # Seed random generator for repeatibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Download data if not present already
    download_and_extract()
    compute_mean_and_std()

    # Make checkpoints subdir if not existing
    os.makedirs("checkpoints", exist_ok=True)
    
    # Make sure we can reach the installed binaries. This is needed for the workspace
    if os.path.exists("/data/DLND/C2/landmark_images"):
        os.environ['PATH'] = f"{os.environ['PATH']}:/root/.local/bin"


def get_data_location():
    """
    Find the location of the dataset, either locally or in the Udacity workspace
    """

    if os.path.exists("landmark_images"):
        data_folder = "landmark_images"
    elif os.path.exists("/data/DLND/C2/landmark_images"):
        data_folder = "/data/DLND/C2/landmark_images"
    else:
        raise IOError("Please download the dataset first")

    return data_folder


def download_and_extract(
    url="https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip",
):
    
    try:
        
        location = get_data_location()
    
    except IOError:
        # Dataset does not exist
        print(f"Downloading and unzipping {url}. This will take a while...")

        with urllib.request.urlopen(url) as resp:

            with ZipFile(BytesIO(resp.read())) as fp:

                fp.extractall(".")

        print("done")
                
    else:
        
        print(
            "Dataset already downloaded. If you need to re-download, "
            f"please delete the directory {location}"
        )
        return None


# Compute image normalization
def compute_mean_and_std():
    """
    Compute per-channel mean and std of the dataset (to be used in transforms.Normalize())
    """

    cache_file = "mean_and_std.pt"
    if os.path.exists(cache_file):
        print(f"Reusing cached mean and std")
        d = torch.load(cache_file)

        return d["mean"], d["std"]

    folder = get_data_location()
    ds = datasets.ImageFolder(
        folder, transform=transforms.Compose([transforms.ToTensor()])
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=0
    )

    mean = 0.0
    for images, _ in tqdm(dl, total=len(ds), desc="Computing mean", ncols=80):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dl.dataset)

    var = 0.0
    npix = 0
    for images, _ in tqdm(dl, total=len(ds), desc="Computing std", ncols=80):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
        npix += images.nelement()

    std = torch.sqrt(var / (npix / 3))

    # Cache results so we don't need to redo the computation
    torch.save({"mean": mean, "std": std}, cache_file)

    return mean, std


def after_subplot(ax: plt.Axes, group_name: str, x_label: str):
    """Add title xlabel and legend to single chart"""
    ax.set_title(group_name)
    ax.set_xlabel(x_label)
    ax.legend(loc="center right")

    if group_name.lower() == "loss":
        ax.set_ylim([None, 4.5])


def plot_confusion_matrix(pred, truth):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np

    gt = pd.Series(truth, name='Ground Truth')
    predicted = pd.Series(pred, name='Predicted')

    confusion_matrix = pd.crosstab(gt, predicted)

    fig, sub = plt.subplots(figsize=(14, 12))
    with sns.plotting_context("notebook"):
        idx = (confusion_matrix == 0)
        confusion_matrix[idx] = np.nan
        sns.heatmap(confusion_matrix, annot=True, ax=sub, linewidths=0.5, linecolor='lightgray', cbar=False)


def visualize_comprehensive_metrics(truth, pred, class_names):
    from sklearn.metrics import classification_report

    """
    Visualize F1, Precision, and Recall together for comprehensive view
    Optimized for large number of classes (50+)
    """
    # Get classification report as dictionary
    report = classification_report(truth, pred, target_names=class_names, output_dict=True)

    # Extract metrics for each class
    metrics_data = []
    for class_name in class_names:
        if class_name in report:
            metrics_data.append({
                'Class': class_name,
                'F1-Score': report[class_name]['f1-score'],
                'Precision': report[class_name]['precision'],
                'Recall': report[class_name]['recall']
            })

    # Create DataFrame
    df = pd.DataFrame(metrics_data)

    # Create subplots: 2 rows, 2 columns
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    # 1. F1-Score horizontal bar plot (main focus)
    bars1 = ax1.barh(range(len(class_names)), df['F1-Score'],
                     color='steelblue', alpha=0.8, height=0.6)
    ax1.set_xlabel('F1-Score')
    ax1.set_ylabel('Classes')
    ax1.set_title('F1-Score per Class', fontsize=14, fontweight='bold')
    ax1.set_yticks(range(len(class_names)))
    ax1.set_yticklabels(class_names, fontsize=8)
    ax1.set_xlim(0, 1)
    ax1.grid(axis='x', alpha=0.3)

    # Add value labels for F1 scores
    for i, (bar, score) in enumerate(zip(bars1, df['F1-Score'])):
        ax1.text(score + 0.01, i, f'{score:.3f}',
                ha='left', va='center', fontsize=7)

    # 2. Precision horizontal bar plot
    bars2 = ax2.barh(range(len(class_names)), df['Precision'],
                     color='forestgreen', alpha=0.8, height=0.6)
    ax2.set_xlabel('Precision')
    ax2.set_ylabel('Classes')
    ax2.set_title('Precision per Class', fontsize=14, fontweight='bold')
    ax2.set_yticks(range(len(class_names)))
    ax2.set_yticklabels(class_names, fontsize=8)
    ax2.set_xlim(0, 1)
    ax2.grid(axis='x', alpha=0.3)

    # Add value labels for precision
    for i, (bar, score) in enumerate(zip(bars2, df['Precision'])):
        ax2.text(score + 0.01, i, f'{score:.3f}',
                ha='left', va='center', fontsize=7)

    # 3. Recall horizontal bar plot
    bars3 = ax3.barh(range(len(class_names)), df['Recall'],
                     color='darkorange', alpha=0.8, height=0.6)
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Classes')
    ax3.set_title('Recall per Class', fontsize=14, fontweight='bold')
    ax3.set_yticks(range(len(class_names)))
    ax3.set_yticklabels(class_names, fontsize=8)
    ax3.set_xlim(0, 1)
    ax3.grid(axis='x', alpha=0.3)

    # Add value labels for recall
    for i, (bar, score) in enumerate(zip(bars3, df['Recall'])):
        ax3.text(score + 0.01, i, f'{score:.3f}',
                ha='left', va='center', fontsize=7)

    # 4. Summary statistics and worst/best performing classes
    ax4.axis('off')

    # Calculate summary statistics
    avg_f1 = df['F1-Score'].mean()
    avg_precision = df['Precision'].mean()
    avg_recall = df['Recall'].mean()

    # Find best and worst performing classes
    best_f1_idx = df['F1-Score'].idxmax()
    worst_f1_idx = df['F1-Score'].idxmin()

    best_class = df.loc[best_f1_idx, 'Class']
    worst_class = df.loc[worst_f1_idx, 'Class']
    best_f1 = df.loc[best_f1_idx, 'F1-Score']
    worst_f1 = df.loc[worst_f1_idx, 'F1-Score']

    # Create summary text
    summary_text = f"""
    PERFORMANCE SUMMARY
    ==================

    Average Metrics:
    • F1-Score: {avg_f1:.3f}
    • Precision: {avg_precision:.3f}
    • Recall: {avg_recall:.3f}

    Best Performing Class:
    • {best_class}: {best_f1:.3f}

    Worst Performing Class:
    • {worst_class}: {worst_f1:.3f}

    Classes with F1 > 0.8: {len(df[df['F1-Score'] > 0.8])}
    Classes with F1 < 0.5: {len(df[df['F1-Score'] < 0.5])}

    Total Classes: {len(class_names)}
    """

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=12, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()
    plt.show()

    # Also return the dataframe for further analysis
    # return df


def get_last_conv_layer(model):
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv
def generate_grad_cam(image_path, model, input_size=(224, 224), target_layer=None):
    from PIL import Image
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # Load & preprocess image
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)

    # Auto-detect target layer if not given
    if target_layer is None:
        target_layer = get_last_conv_layer(model)

    activations, gradients = [], []
    pred_class = None

    def forward_hook(module, input, output):
        activations.clear()
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    # Register hooks
    hook_fwd = target_layer.register_forward_hook(forward_hook)
    hook_bwd = target_layer.register_backward_hook(backward_hook)

    model.eval()
    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    # Backprop for predicted class
    model.zero_grad()
    output[0, pred_class].backward()

    act = activations[0].squeeze().detach().cpu()
    grad = gradients[0].squeeze().detach().cpu()

    # Global average pooling of gradients
    weights = grad.mean(dim=(1, 2))
    cam = torch.zeros(act.shape[1:], dtype=torch.float32)

    for i, w in enumerate(weights):
        cam += w * act[i]

    # Normalize CAM
    cam = np.maximum(cam.numpy(), 0)
    cam = cv2.resize(cam, input_size)
    cam = (cam - cam.min()) / (cam.max() + 1e-8)

    # Clean hooks
    hook_fwd.remove()
    hook_bwd.remove()

    return img, cam, pred_class


def plot_grad_cam(img, cam):
    img_array = np.array(img.resize((224, 224)))
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.5, heatmap, 0.5, 0)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title("Original")

    plt.subplot(1, 3, 2)
    plt.imshow(cam, cmap='jet')
    plt.title("Grad-CAM")

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    # plt.title(f"Overlay - {class_names[pred_class]}")
    plt.title(f"Overlay")
    plt.tight_layout()
    plt.show()


