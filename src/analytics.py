import pandas as pd,matplotlib.pyplot as plt
def plot_blob_stats():
    df=pd.read_csv("../results/blob_features.csv"); fstats=pd.read_csv("../results/flow_stats.csv")
    plt.figure(figsize=(8,4)); plt.subplot(1,2,1)
    for bid,g in df.groupby("BlobID"): plt.plot(g["Frame"],g["Area"],label=f"Blob {bid}")
    plt.legend(); plt.title("Blob Area over Time")
    plt.subplot(1,2,2)
    plt.plot(fstats["Frame"],fstats["AvgFlow"],'r-'); plt.title("Avg Optical Flow per Frame")
    plt.tight_layout(); plt.savefig("../results/blob_flow_stats.png"); plt.close()
