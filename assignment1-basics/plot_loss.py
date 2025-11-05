# 根据训练和验证损失绘制损失曲线
import matplotlib.pyplot as plt
import pandas as pd

csv_path = "./out/错误数据集上的结果/log.csv"
output_path = "./out/plot.png"

# 只保留 val_loss 不为空的行
df = pd.read_csv(csv_path)
df = df.dropna(subset=["val_loss"])
plt.figure(figsize=(20, 12))
plt.plot(df["step"], df["train_loss"], label="Train Loss", linewidth=1.5)
plt.plot(df["step"], df["val_loss"], label="Validation Loss", linewidth=1.5)
plt.axhline(y=1.45, color="r", linestyle="--", linewidth=1.2, label="Threshold = 1.45")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training and Validation Loss (Non-empty Val)")
plt.legend()
plt.grid()
plt.savefig(output_path)
plt.close()