import numpy as np
import cv2
import os
import tensorflow as tf
import nibabel as nib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA, NMF
from skimage import restoration
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import random
from scipy.signal.windows import gaussian
from scipy.ndimage import convolve
from scipy import sparse
from scipy.sparse.linalg import eigsh
import pywt

np.random.seed(42)  # Set NumPy seed
random.seed(42)  # Set random seed
tf.random.set_seed(42)  # Set TensorFlow seed

class ImageProcessing:
    def __init__(self, use_pretrained=False, use_subpixel=False, use_spectral_unmixing=False, use_decomposition=False):
        self.use_pretrained = use_pretrained
        self.use_subpixel = use_subpixel
        self.use_spectral_unmixing = use_spectral_unmixing
        self.use_decomposition = use_decomposition
        self.skull_stripper = SkullStripping(use_pretrained)

    def spectral_unmixing(self, image):
        """
        Perform spectral unmixing using Non-Negative Matrix Factorization (NMF).
        """
        if self.use_spectral_unmixing:
            if len(image.shape) == 2:  # If grayscale, replicate channels
                image = np.stack([image] * 3, axis=-1)

            # Ensure non-negative values
            image = np.maximum(image, 0)
            
            # Flatten the spectral image for processing
            reshaped_image = image.reshape((-1, image.shape[-1]))
            
            # Add small constant to avoid zeros
            epsilon = 1e-10
            reshaped_image = reshaped_image + epsilon

            # Apply NMF for spectral unmixing
            n_components = 3  # Adjust this based on the number of endmembers
            nmf = NMF(n_components=n_components, init='random', random_state=42, max_iter=500)
            W = nmf.fit_transform(reshaped_image)  # Abundances
            H = nmf.components_  # Endmember signatures

            # Reconstruct the most relevant abundance map for visualization
            abundance_map = W[:, 0].reshape(image.shape[:2])  # Use the first component
            
            # Normalize the abundance map
            abundance_map = (abundance_map - abundance_map.min()) / (abundance_map.max() - abundance_map.min())
            abundance_map = (abundance_map * 255).astype(np.uint8)
            
            return abundance_map

        return image

    def subpixel_mapping(self, image):
        """
        Improved subpixel mapping with customizable parameters and optimization.
        """
        if not self.use_subpixel:
            return image

        # Initialize parameters for simulated annealing
        initial_temp = 150  # Increased initial temperature
        final_temp = 0.01   # Lower final temperature for more fine-grained search
        cooling_rate = 0.90  # Slower cooling rate
        iterations_per_temp = 50  # Fewer iterations per temperature step

        # Create high-resolution grid (2x upscaling)
        scale_factor = 2
        h, w = image.shape[:2]
        hr_height, hr_width = h * scale_factor, w * scale_factor
        hr_image = np.zeros((hr_height, hr_width))

        def calculate_spatial_dependence(pos, neighbors):
            """Calculate spatial dependency score for a position."""
            x, y = pos
            score = 0
            for nx, ny in neighbors:
                if 0 <= nx < hr_height and 0 <= ny < hr_width:
                    dist = np.sqrt((x - nx)**2 + (y - ny)**2)
                    score += float(hr_image[nx, ny]) / (dist + 1e-6)
            return score

        def calculate_directivity(pos, window_size=5):
            """Calculate directivity score using Sobel operators with larger window."""
            x, y = pos
            if x < window_size or y < window_size or x >= hr_height-window_size or y >= hr_width-window_size:
                return 0.0

            window = hr_image[x-window_size:x+window_size+1, y-window_size:y+window_size+1]
            sobel_x = cv2.Sobel(window, cv2.CV_64F, 1, 0, ksize=5)
            sobel_y = cv2.Sobel(window, cv2.CV_64F, 0, 1, ksize=5)
            gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

            mean_direction = np.arctan2(np.mean(sobel_y), np.mean(sobel_x))
            return float(np.mean(gradient_magnitude) * np.cos(mean_direction))

        def calculate_connectivity(pos):
            """Calculate connectivity score based on local neighborhood."""
            x, y = pos
            neighborhood = [
                (x-1, y), (x+1, y),
                (x, y-1), (x, y+1)
            ]
            return float(sum(1 for nx, ny in neighborhood
                            if 0 <= nx < hr_height and 0 <= ny < hr_width and hr_image[nx, ny] > 0))

        def calculate_energy(pos):
            """Calculate total energy for a position."""
            neighbors = [
                (pos[0]-1, pos[1]), (pos[0]+1, pos[1]),
                (pos[0], pos[1]-1), (pos[0], pos[1]+1)
            ]
            spatial_score = calculate_spatial_dependence(pos, neighbors)
            directivity_score = calculate_directivity(pos)
            connectivity_score = calculate_connectivity(pos)
            return float(-(spatial_score + 0.4 * directivity_score + 0.3 * connectivity_score))

        # Initial allocation using Lanczos interpolation
        hr_image = cv2.resize(image, (hr_width, hr_height), interpolation=cv2.INTER_LANCZOS4)

        # Simulated annealing optimization
        current_temp = initial_temp
        while current_temp > final_temp:
            for _ in range(iterations_per_temp):
                x = np.random.randint(1, hr_height-1)
                y = np.random.randint(1, hr_width-1)
                pos = (x, y)

                current_energy = calculate_energy(pos)
                original_value = float(hr_image[x, y])
                hr_image[x, y] = np.random.normal(original_value, 0.05)

                new_energy = calculate_energy(pos)
                energy_diff = float(new_energy - current_energy)
                acceptance_probability = np.exp(-energy_diff / current_temp)

                if energy_diff < 0 or np.random.random() < acceptance_probability:
                    continue
                else:
                    hr_image[x, y] = original_value

            current_temp *= cooling_rate

        hr_image = np.clip(hr_image, 0, 255)
        enhanced_image = cv2.resize(hr_image, (w, h), interpolation=cv2.INTER_AREA)

        return enhanced_image.astype(np.uint8)
    def wavelength_decomposition(self, image):
      """
      Memory-efficient spectral graph wavelet transform using block processing
      and simplified calculations.
      """
      if not self.use_decomposition:
          return image

      def create_block_laplacian(block, radius=1):
          """Create a local graph Laplacian for a small block."""
          h, w = block.shape
          n = h * w
          
          # Create sparse adjacency matrix for local block
          indices = []
          values = []
          
          for i in range(h):
              for j in range(w):
                  current_idx = i * w + j
                  current_val = float(block[i, j])
                  
                  # Connect to immediate neighbors only
                  for di in [-1, 0, 1]:
                      for dj in [-1, 0, 1]:
                          if di == 0 and dj == 0:
                              continue
                          ni, nj = i + di, j + dj
                          if 0 <= ni < h and 0 <= nj < w:
                              neighbor_idx = ni * w + nj
                              weight = np.exp(-abs(current_val - float(block[ni, nj])) / 10.0)
                              indices.append([current_idx, neighbor_idx])
                              values.append(weight)
          
          if not indices:  # Handle empty blocks
              return None, None
              
          indices = np.array(indices)
          values = np.array(values)
          
          # Create sparse adjacency matrix
          A = sparse.coo_matrix((values, (indices[:, 0], indices[:, 1])), 
                              shape=(n, n)).tocsr()
          
          # Compute degree matrix
          degrees = np.array(A.sum(axis=1)).flatten()
          D_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees + 1e-10))
          
          # Compute normalized Laplacian
          L = sparse.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
          return L, degrees

      def process_block(block, scale=2.0):
          """Process a single block of the image."""
          if block.size == 0:
              return np.zeros_like(block)
              
          # Create local Laplacian
          L, degrees = create_block_laplacian(block)
          if L is None:
              return block
              
          # Compute a few eigenvalues/vectors (reduced number)
          try:
              k = min(5, L.shape[0] - 1)  # Use fewer eigenvalues
              eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')
              
              # Simple wavelet transform
              transformed = np.zeros_like(block, dtype=float)
              block_flat = block.flatten()
              
              # Apply single-scale transform
              kernel = np.exp(-scale * eigenvalues)
              wavelet = eigenvectors @ sparse.diags(kernel) @ eigenvectors.T
              transformed_flat = wavelet @ block_flat
              
              transformed = transformed_flat.reshape(block.shape)
              transformed = np.clip(transformed, 0, 255)
              
              return transformed
              
          except:
              return block

      # Convert image to float
      image_float = image.astype(float)
      
      # Handle different channel configurations
      if len(image_float.shape) == 3:
          h, w, c = image_float.shape
          processed = np.zeros_like(image_float)
      else:
          h, w = image_float.shape
          c = 1
          image_float = image_float.reshape(h, w, 1)
          processed = np.zeros((h, w, 1))

      # Process image in blocks
      block_size = 32  # Smaller block size
      overlap = 4     # Small overlap to reduce boundary effects

      for channel in range(c):
          channel_data = image_float[:, :, channel]
          processed_channel = np.zeros_like(channel_data)
          
          # Process blocks with overlap
          for i in range(0, h, block_size - overlap):
              for j in range(0, w, block_size - overlap):
                  # Extract block
                  block_end_i = min(i + block_size, h)
                  block_end_j = min(j + block_size, w)
                  block = channel_data[i:block_end_i, j:block_end_j]
                  
                  # Process block
                  processed_block = process_block(block)
                  
                  # Handle overlap
                  if i == 0:
                      start_i = 0
                  else:
                      start_i = overlap // 2
                      
                  if j == 0:
                      start_j = 0
                  else:
                      start_j = overlap // 2
                      
                  if block_end_i == h:
                      end_i = block_end_i - i
                  else:
                      end_i = block_size - overlap // 2
                      
                  if block_end_j == w:
                      end_j = block_end_j - j
                  else:
                      end_j = block_size - overlap // 2
                  
                  processed_channel[i + start_i:i + end_i, 
                                  j + start_j:j + end_j] = processed_block[start_i:end_i, 
                                                                        start_j:end_j]
          
          processed[:, :, channel] = processed_channel

      # Normalize output
      processed = np.clip(processed, 0, 255)
      
      if c == 1:
          processed = processed.reshape(h, w)
          
      return processed.astype(image.dtype)
    

    def reduce_channels(self, image, n_components=3):
        """
        Reduce the number of channels in the image using PCA.
        """
        if len(image.shape) == 2:
            return image

        reshaped_image = image.reshape((-1, image.shape[-1]))
        n_components = min(n_components, reshaped_image.shape[1])
        pca = PCA(n_components=n_components)
        reduced_image = pca.fit_transform(reshaped_image)
        reduced_image = reduced_image.reshape(image.shape[0], image.shape[1], n_components)

        if reduced_image.shape[-1] != n_components:
            reduced_image = reduced_image[:, :, :n_components]

        return reduced_image

    def process_image(self, image):
        """
        Process the image using skull stripping, subpixel mapping, spectral unmixing, and wavelength decomposition.
        """
        if self.use_pretrained:
            image = self.skull_stripper.pretrained_skull_strip(image)
        else:
            image = self.skull_stripper.advanced_skull_strip(image)

        image = self.subpixel_mapping(image)
        image = self.spectral_unmixing(image)
        image = self.wavelength_decomposition(image)

        if len(image.shape) > 2 and image.shape[-1] > 3:
            image = self.reduce_channels(image)

        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)

        return image



class SkullStripping:
    def __init__(self, use_pretrained=False):
        self.use_pretrained = use_pretrained

    def traditional_skull_strip(self, mri_image):
        """
        Improved traditional skull stripping using adaptive thresholding and elliptical kernels.
        """
        if len(mri_image.shape) > 2:
            gray = cv2.cvtColor(mri_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = mri_image

        # Denoising
        denoised = restoration.denoise_nl_means(gray, h=1.15 * np.std(gray))
        denoised = (denoised * 255).astype(np.uint8)

        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Find and filter contours
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(gray.shape, dtype=np.uint8)

        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Filter small areas
                cv2.drawContours(mask, [contour], -1, 255, -1)

        stripped_image = cv2.bitwise_and(gray, gray, mask=mask)
        return stripped_image

    def advanced_skull_strip(self, mri_image):
        """
        Improved advanced skull stripping using Laplacian of Gaussian (LoG) for edge detection.
        """
        if len(mri_image.shape) > 2:
            gray = cv2.cvtColor(mri_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = mri_image

        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Edge detection using LoG
        log_edges = cv2.Laplacian(enhanced, cv2.CV_64F)
        log_edges = np.uint8(np.absolute(log_edges))

        # Morphological closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed_edges = cv2.morphologyEx(log_edges, cv2.MORPH_CLOSE, kernel)

        # Find and filter contours
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        stripped_image = cv2.bitwise_and(gray, gray, mask=mask)
        return stripped_image

    def pretrained_skull_strip(self, mri_image_path):
        """
        Use a pre-trained skull stripping tool such as FSL BET or HD-BET.
        """
        # Placeholder: Replace with actual integration code for FSL or HD-BET.
        print(f"Using pre-trained model for {mri_image_path}.")
        return cv2.imread(mri_image_path, cv2.IMREAD_GRAYSCALE)


class TumorDetectionPreprocessor:
    def __init__(self, yes_path, no_path, use_skull_stripping=True, use_pretrained=False, use_subpixel=False, use_spectral_unmixing=False, use_decomposition=False):
        self.yes_path = yes_path
        self.no_path = no_path
        self.use_skull_stripping = use_skull_stripping
        self.image_processor = ImageProcessing(use_pretrained, use_subpixel, use_spectral_unmixing, use_decomposition)

    def load_and_split_data(self, test_size=0.2, random_state=42):
        tumor_images = self._load_images(self.yes_path, label=1)
        non_tumor_images = self._load_images(self.no_path, label=0)
        all_images = tumor_images + non_tumor_images
        np.random.shuffle(all_images)

        X = [item[0] for item in all_images]
        y = [item[1] for item in all_images]

        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    def _load_images(self, path, label):
        processed_images = []
        for filename in os.listdir(path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.nii')):
                full_path = os.path.join(path, filename)
                if filename.lower().endswith('.nii'):
                    try:
                        nifti_img = nib.load(full_path)
                        image = nifti_img.get_fdata()
                        if len(image.shape) == 3:
                            image = image[:, :, image.shape[2] // 2]
                        image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
                        continue
                else:
                    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

                image = cv2.resize(image, (224, 224))  # Resize to 224x224

                # Process the image
                processed_image = self.image_processor.process_image(image)
                processed_images.append((processed_image, label))

        return processed_images


class TumorDetectionModel:
    def __init__(self):
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(1, activation='sigmoid')(x)
        self.model = Model(inputs=base_model.input, outputs=x)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        history = self.model.fit(np.array(X_train), np.array(y_train), validation_data=(np.array(X_val), np.array(y_val)),
                       epochs=epochs, batch_size=batch_size)

        # Training metrics (Accuracy, Precision, Recall, F1 Score)
        y_train_pred = (self.model.predict(np.array(X_train)) > 0.5).astype(int)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)

        print(f"Training Accuracy: {train_accuracy}")
        print(f"Training Precision: {train_precision}")
        print(f"Training Recall: {train_recall}")
        print(f"Training F1 Score: {train_f1}")

        return history

    def evaluate(self, X_test, y_test):
        y_pred = (self.model.predict(np.array(X_test)) > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"Test Accuracy: {accuracy}")
        print(f"Test Precision: {precision}")
        print(f"Test Recall: {recall}")
        print(f"Test F1 Score: {f1}")
        print(f"Confusion Matrix:\n{cm}")

        return accuracy, precision, recall, f1, cm


def main(yes_path, no_path, use_skull_stripping=True, use_pretrained=False, use_subpixel=True, use_spectral_unmixing=True, use_decomposition=True):
    preprocessor = TumorDetectionPreprocessor(yes_path, no_path, use_skull_stripping, use_pretrained, use_subpixel, use_spectral_unmixing, use_decomposition)
    X_train, X_test, y_train, y_test = preprocessor.load_and_split_data()

    model = TumorDetectionModel()
    model.train(X_train, y_train, X_test, y_test, epochs=5, batch_size=32)
    accuracy, precision, recall, f1, cm = model.evaluate(X_test, y_test)


# Example usage
yes_path = "/media/yes"  # Replace with actual path
no_path = "/media/no"  # Replace with actual path
main(yes_path, no_path, use_skull_stripping=True, use_pretrained=False, use_subpixel=True, use_spectral_unmixing=True, use_decomposition=True)
