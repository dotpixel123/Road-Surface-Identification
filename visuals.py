import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

train_set = "Dataset\\Data\\train"
val_test = "Dataset\\Data\\val"
test_set = "Dataset\\Data\\test"

class_names = [
    "dry-asphalt-smooth", "dry-concrete-smooth", "dry-gravel", "dry-mud",
    "fresh_snow", "ice", "melted_snow", "water-asphalt-smooth",
    "water-concrete-smooth", "water-mud", "wet-asphalt-smooth",
    "wet-concrete-smooth", "wet-gravel", "wet-mud"
]
class_mapping = {name: idx for idx, name in enumerate(class_names)}


class_counts = Counter()

# Count images per class
for img in os.listdir(train_set):
    for class_name in class_names:
        if class_name in img:  # Check if class name is in the image filename
            class_counts[class_name] += 1
            break  

print(class_counts)

classes, counts = zip(*class_counts.items())    

print(counts)

plt.figure(figsize=(12, 6))
sns.barplot(x=list(classes), y=list(counts), palette="viridis")
plt.xticks(rotation=45, ha="right")  # Rotate class names for better visibility
plt.xlabel("Class Names")
plt.ylabel("Number of Images")
plt.title("Distribution of Images in Each Class (Test Set)")
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Show the chart
plt.show()