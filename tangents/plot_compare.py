import numpy as np
import matplotlib.pyplot as plt


def plot_comparison(original, decoded, sr, start_s=0.2, window_s=0.02, save_path="comparison.png"):
    """Plot a small window of the original audio, the decoded audio, and both overlaid."""
    # zoom on a small window (the whole signal would just look like a solid block)
    start = int(start_s * sr)
    end = start + int(window_s * sr)
    window_ms = np.arange(start, end) / sr * 1000

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(window_ms, original[start:end], color="tab:blue")
    axes[0].set_title("Original")

    axes[1].plot(window_ms, decoded[start:end], color="tab:orange")
    axes[1].set_title("Decoded (after going through the codec)")

    axes[2].plot(window_ms, original[start:end], color="tab:blue", label="original")
    axes[2].plot(window_ms, decoded[start:end], color="tab:orange", alpha=0.7, label="decoded")
    axes[2].set_title("Both on top of each other")
    axes[2].legend()

    axes[2].set_xlabel("Time (ms)")
    for ax in axes:
        ax.set_ylabel("Amplitude")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
