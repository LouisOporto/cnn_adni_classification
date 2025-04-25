import os
import zipfile
import kagglehub
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import DenseNet169
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# code from dataset website to load and train the model
# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
# Check if GPU is available
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow version:", tf.__version__)
def extract_dataset(zip_path, extract_path):
    """Extract the dataset from a ZIP file"""
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Dataset extracted to {extract_path}")
download_dir = "my_dataset_folder"
path = kagglehub.dataset_download("lukechugh/best-alzheimer-mri-dataset-99-accuracy",download_dir=download_dir)
print("Path to dataset files:", path)
for file in os.listdir(path):
    if file.endswith(".zip"):
        zip_path = os.path.join(path, file)
        extract_path = os.path.join(path, "extracted")
        extract_dataset(zip_path, extract_path)
        break 
def explore_dataset(data_dir):
    """Explore the dataset structure and count samples"""
    try:
        class_names = sorted(os.listdir(data_dir))
        class_names = [c for c in class_names if os.path.isdir(os.path.join(data_dir, c))]

        if not class_names:
            print(f"ERROR: No class directories found in {data_dir}")
            print(f"Contents of {data_dir}: {os.listdir(data_dir)}")
            return None, None

        print(f"Classes: {class_names}")

        total_samples = 0
        class_counts = {}

        for class_name in class_names:
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
                count = len(files)
                class_counts[class_name] = count
                total_samples += count
                print(f"Class {class_name}: {count} images")

                # Check first few images to ensure they can be loaded
                for img_file in files[:3]:
                    try:
                        img_path = os.path.join(class_dir, img_file)
                        img = load_img(img_path)
                        arr = img_to_array(img)
                        print(f"  Sample image shape: {arr.shape}, min: {arr.min()}, max: {arr.max()}")
                    except Exception as e:
                        print(f"  ERROR loading {img_path}: {str(e)}")

        print(f"Total samples: {total_samples}")

        return class_names, class_counts
    except Exception as e:
        print(f"Error exploring dataset: {str(e)}")
        return None, None
# Function to visualize sample images with improved error handling
def visualize_samples(data_dir, class_names, samples_per_class=2):
    """Visualize sample images from each class"""
    plt.figure(figsize=(15, 10))

    for i, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))][:samples_per_class]

        for j, img_file in enumerate(image_files):
            try:
                img_path = os.path.join(class_dir, img_file)
                img = plt.imread(img_path)

                plt.subplot(len(class_names), samples_per_class, i*samples_per_class + j + 1)
                plt.imshow(img, cmap='gray')
                plt.title(f"{class_name}\n{img.shape}" if j == 0 else f"{img.shape}")
                plt.axis('off')

            except Exception as e:
                print(f"Error visualizing {img_file}: {str(e)}")

    plt.tight_layout()
    plt.show()
# Function to create an improved model for distinguishing subtle differences in MRI scans
def create_improved_multiclass_model(input_shape=(256, 256, 3), num_classes=4, use_grayscale=False):
    """Create a model specifically designed to distinguish between similar MRI classes"""
    # Adjust input shape for grayscale
    if use_grayscale:
        input_shape = (input_shape[0], input_shape[1], 1)

    # Create input layer
    inputs = Input(shape=input_shape)

    # Convert grayscale to 3 channels if needed
    if use_grayscale:
        x = tf.keras.layers.Conv2D(3, (1, 1))(inputs)
    else:
        x = inputs

    # Use DenseNet169 which has more parameters than DenseNet121
    base_model = DenseNet169(
        weights='imagenet',
        include_top=False,
        input_shape=(input_shape[0], input_shape[1], 3)
    )

    # Freeze early layers
    for layer in base_model.layers[:400]:
        layer.trainable = False

    # Make later layers trainable from the start
    for layer in base_model.layers[400:]:
        layer.trainable = True

    x = base_model(x)

    # Global pooling
    x = GlobalAveragePooling2D()(x)

    # Add more capacity with deeper layers
    x = BatchNormalization()(x)
    x = Dense(1024, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    # Add another layer with more capacity
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    # Add a layer focused on learning subtle features
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Final classification layer
    outputs = Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    # Custom learning rate
    optimizer = Adam(learning_rate=0.0002)

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(multi_label=True)]
    )

    return model
def plot_training_history(history):
    """Plot the training and validation metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    ax1.grid(True)

    # Plot training & validation loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
# Function to evaluate the model
def evaluate_multiclass_model(model, test_generator, class_names):
    """Evaluate model performance and visualize results for multi-class classification"""
    # Reset the test generator
    test_generator.reset()

    # Get predictions
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)

    # Get true labels - need to convert from one-hot back to class indices
    y_true = np.argmax(test_generator.labels, axis=1) if len(test_generator.labels.shape) > 1 else test_generator.labels

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Per-class accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    for i, accuracy in enumerate(class_accuracy):
        print(f"Class {class_names[i]} accuracy: {accuracy:.4f}")

    # Create a bar chart of class accuracies
    plt.figure(figsize=(12, 6))
    plt.bar(class_names, class_accuracy, color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Plot confusion examples for the worst performing class
    worst_class_idx = np.argmin(class_accuracy)
    worst_class_name = class_names[worst_class_idx]
    print(f"\nAnalyzing confusion for worst performing class: {worst_class_name}")

    # Find indices of misclassified samples for the worst class
    misclassified_indices = np.where((y_true == worst_class_idx) & (y_pred != worst_class_idx))[0]

    # Show confusion examples if any exist
    if len(misclassified_indices) > 0:
        print(f"Found {len(misclassified_indices)} misclassified samples for class {worst_class_name}")

        # Reset the generator to get images
        test_generator.reset()
        all_images = []
        all_true = []
        for _ in range(len(test_generator)):
            images, labels = next(test_generator)
            all_images.append(images)
            all_true.append(labels)
            if len(all_images) * images.shape[0] >= len(y_true):
                break

        all_images = np.vstack(all_images[:len(test_generator)])

        # Plot a few examples of confusion
        n_examples = min(8, len(misclassified_indices))
        plt.figure(figsize=(16, 8))
        for i in range(n_examples):
            idx = misclassified_indices[i]
            img = all_images[idx]
            true_class = class_names[y_true[idx]]
            pred_class = class_names[y_pred[idx]]

            plt.subplot(2, 4, i+1)
            if img.shape[-1] == 1:  # Grayscale
                plt.imshow(img[:,:,0], cmap='gray')
            else:
                plt.imshow(img)
            plt.title(f"True: {true_class}\nPred: {pred_class}", fontsize=10)
            plt.axis('off')

        plt.tight_layout()
        plt.show()
# Main execution
def main():
    # Set paths for your dataset
    zip_path = '/content/alz.zip'
    extract_path = '/content'
    train_dir = os.path.join(extract_path, 'alz/train')

    # Extract dataset if needed
    if not os.path.exists(train_dir):
        print("Extracting dataset...")
        extract_dataset(zip_path, extract_path)

    # Verify dataset structure
    print("\nVerifying dataset structure:")
    print(f"Extract path contents: {os.listdir(extract_path)}")

    # Explore dataset
    print("\nExploring dataset...")
    class_names, class_counts = explore_dataset(train_dir)

    if not class_names:
        print("ERROR: Could not properly identify classes. Please check dataset structure.")
        return

    # Visualize samples
    print("\nVisualizing sample images...")
    visualize_samples(train_dir, class_names)

    # Create class weights with emphasis on problematic classes
    class_indices = {name: idx for idx, name in enumerate(class_names)}

    # Create custom class weights with greater emphasis on "Very Mild Impairment"
    class_weights = {}
    for name, idx in class_indices.items():
        if "Very Mild Impairment" in name:
            class_weights[idx] = 7.0  # Significantly higher weight for the most problematic class
        elif "No Impairment" in name:
            class_weights[idx] = 2.0  # Higher weight for No Impairment which had poor precision
        else:
            class_weights[idx] = 1.0  # Normal weight for other classes

    print(f"Using class weights: {class_weights}")

    # Data augmentation specifically for MRI images
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        brightness_range=[0.9, 1.1],  # Slight brightness variation
        horizontal_flip=True,
        fill_mode='constant',
        cval=0,  # Black background for padding
        validation_split=0.2
    )

    # Test data generator (only rescaling)
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    # Image dimensions - use higher resolution
    img_height, img_width = 256, 256  # Higher resolution for better feature detection
    batch_size = 16

    # Use grayscale since MRI images are grayscale
    color_mode = 'grayscale'

    # Create train generator
    print(f"\nCreating training generator with color_mode={color_mode}...")
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        color_mode=color_mode,
        shuffle=True
    )

    # Create validation generator
    validation_generator = test_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        color_mode=color_mode,
        shuffle=False
    )

    # Print class indices
    print("\nClass indices:", train_generator.class_indices)

    # Create improved model
    print("\nCreating improved 4-class model...")
    input_shape = (img_height, img_width, 1)  # Grayscale
    model = create_improved_multiclass_model(input_shape=input_shape, num_classes=len(class_names), use_grayscale=True)
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(
        'alzheimers_model_4class_improved.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,  # Still generous patience
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # More gentle reduction
        patience=7,
        min_lr=1e-7,
        verbose=1
    )

    # Train the model with weighted classes
    print("\nTraining the improved 4-class model with weighted classes...")
    epochs = 40  # Reasonable number of epochs with early stopping

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint, early_stopping, reduce_lr],
        class_weight=class_weights,  # Apply class weights
        verbose=1
    )

    # Plot training history
    plot_training_history(history)

    # Evaluate the improved model
    print("\nEvaluating the improved model...")
    model.load_weights('alzheimers_model_4class_improved.h5')
    evaluate_multiclass_model(model, validation_generator, list(train_generator.class_indices.keys()))

    # Save the final model
    model.save('alzheimers_classification_model_4class_improved.h5')
if __name__ == "__main__":
    main()