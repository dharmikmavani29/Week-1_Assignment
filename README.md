# ♻️ AICTE Internship - E-Waste Image Classification using TensorFlow

## 📅 Week 1: Dataset Setup, Preprocessing & Visualization (21st June 2025)

---

## 1. 📁 Dataset Loading

Defined dataset paths and loaded training, validation, and test sets using `image_dataset_from_directory`.

```python
trainpath = r'C:\Users\admin\Desktop\AICTE INTERNSHIP\modified-dataset\train'
validpath = r'C:\Users\admin\Desktop\AICTE INTERNSHIP\modified-dataset\val'
testpath  = r'C:\Users\admin\Desktop\AICTE INTERNSHIP\modified-dataset\test'

datatrain = tf.keras.utils.image_dataset_from_directory(
    trainpath, shuffle=True, image_size=(128, 128), batch_size=32
)

datavalid = tf.keras.utils.image_dataset_from_directory(
    validpath, shuffle=True, image_size=(128, 128), batch_size=32
)

datatest = tf.keras.utils.image_dataset_from_directory(
    testpath, shuffle=False, image_size=(128, 128), batch_size=32
)
```

---

## 2. 🧠 Class Names

Extracted class names and confirmed class count:

```python
class_names = datatrain.class_names
print(len(class_names))
print(class_names)
```

---

## 3. 📊 Class Distribution Analysis

Created a custom function to count and plot the number of images per class for any dataset.

```python
def plot_class_distribution(dataset, title="Class Distribution"):
    class_counts = {}
    for images, labels in dataset:
        for label in labels.numpy():
            class_name = dataset.class_names[label]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    class_names = list(class_counts.keys())
    counts = list(class_counts.values())

    plt.figure(figsize=(8, 6))
    plt.bar(class_names, counts, color='r', edgecolor='k')
    plt.xlabel("Class")
    plt.ylabel("Number of Items")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
```

✅ Applied this function to:
```python
plot_class_distribution(datatrain, "Training Data Distribution")
plot_class_distribution(datavalid, "Validation Data Distribution")
plot_class_distribution(datatest, "Test Data Distribution")
```

---

## 4. 🖼️ Image Visualization

Displayed 12 sample images from the training set with class names:

```python
plt.figure(figsize=(12, 12))
for images, labels in datatrain.take(1):
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
```

---

## 5. 🔁 Data Augmentation Pipeline

Used Keras augmentation layers to enhance training data dynamically during training:

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])
```

✅ Benefits:
- Randomly flips, rotates, and zooms images
- Improves model robustness to variations in orientation and scale

---

## ✅ Summary of Week 1:
- Loaded and structured the dataset (Train/Validation/Test)
- Verified class names and image counts
- Visualized training samples and class distribution
- Built a real-time data augmentation pipeline

---

