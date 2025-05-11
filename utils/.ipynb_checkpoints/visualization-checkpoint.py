import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(metrics_df, row='RSI_Threshold', col='Holding_Days', value='Sharpe Ratio'):
    pivot_table = metrics_df.pivot(index=row, columns=col, values=value).astype(float)
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': value})
    plt.title(f"{value} Heatmap\n{row} vs {col}")
    plt.ylabel(row)
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()