# Python in Markdown

This document includes Python code executed with pandoc-pyplot.

## Data Visualization

Below is a Python plot of a simple dataset.

```{.python .pandoc-pyplot caption="Sample Plot" output="plot.png"}
import matplotlib.pyplot as plt
import pandas as pd

# Sample data
df = pd.DataFrame({
    'x': [1, 2, 3, 4],
    'y': [10, 20, 25, 30]
})

# Plot
plt.plot(df['x'], df['y'])
plt.title('Linear Trend')
plt.xlabel('X')
plt.ylabel('Y')