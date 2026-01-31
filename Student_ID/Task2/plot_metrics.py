import matplotlib.pyplot as plt
import os

def plot_metrics():
    # Use absolute path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, 'training_log.txt')
    
    if not os.path.exists(log_file):
        print(f"Error: {log_file} not found.")
        return

    epochs = []
    losses = []
    micro_ps = []
    micro_rs = []
    micro_f1s = []

    try:
        with open(log_file, 'r') as f:
            # Read header
            header = f.readline().strip().split('\t')
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                if len(parts) < 5:
                    continue
                
                epochs.append(int(parts[0]))
                losses.append(float(parts[1]))
                micro_ps.append(float(parts[2]))
                micro_rs.append(float(parts[3]))
                micro_f1s.append(float(parts[4]))
    except Exception as e:
        print(f"Error reading log file: {e}")
        return

    if not epochs:
        print("No data found in log file.")
        return

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, marker='o', label='Total Loss', color='tab:red')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss_curve.png')
    print("Saved loss_curve.png")
    plt.close()

    # Plot Micro Metrics
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, micro_ps, marker='s', label='Micro Precision', linestyle='--', color='tab:blue')
    plt.plot(epochs, micro_rs, marker='^', label='Micro Recall', linestyle='--', color='tab:green')
    plt.plot(epochs, micro_f1s, marker='*', label='Micro F1', linewidth=2, color='tab:orange')
    
    plt.title('Micro Metrics on Test Set')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    
    # Adjust y-axis to show details if variance is small, but keep it reasonable
    min_val = min(min(micro_ps), min(micro_rs), min(micro_f1s))
    max_val = max(max(micro_ps), max(micro_rs), max(micro_f1s))
    margin = (max_val - min_val) * 0.1 if max_val != min_val else 0.05
    plt.ylim(max(0, min_val - margin), min(1.0, max_val + margin))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('metrics_curve.png')
    print("Saved metrics_curve.png")
    plt.close()

if __name__ == '__main__':
    plot_metrics()
