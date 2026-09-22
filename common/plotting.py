import matplotlib.pyplot as plt
import os

def plot_training_curves(history, save_path="report/figures/training_curve.png"):
    """
    Plots training loss and validation F1 scores over epochs.
    history: dict containing 'loss', 'mean_f1', and 'domains' metrics.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    epochs = range(1, len(history['loss']) + 1)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot Training Loss on the primary y-axis
    color_loss = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color=color_loss)
    ax1.plot(epochs, history['loss'], color=color_loss, marker='o', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)
    
    # Plot F1 Scores on a secondary y-axis
    ax2 = ax1.twinx()  
    color_f1 = 'tab:blue'
    ax2.set_ylabel('Macro-F1 Score', color=color_f1)
    
    # Plot Mean F1 (bold line)
    ax2.plot(epochs, history['mean_f1'], color='tab:blue', marker='s', 
             linestyle='-', linewidth=2, label='Mean Source F1')
    
    # Plot individual domain F1s (dashed lines)
    domain_colors = ['tab:green', 'tab:orange', 'tab:purple']
    if 'domains' in history:
        for i, (dom, scores) in enumerate(history['domains'].items()):
            ax2.plot(epochs, scores, color=domain_colors[i % len(domain_colors)], 
                     marker='^', linestyle='--', alpha=0.7, label=f'{dom.capitalize()} F1')
            
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    plt.title('Training Loss and Source Validation F1')
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  [*] Training curve saved to {save_path}")