import numpy as np
import os
from PIL import Image
import pickle

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp = np.exp(x - np.max(x))
    return exp / np.sum(exp, axis=-1, keepdims=True)

def max_pool(x, size=2, stride=2):
    n, h, w, c = x.shape
    h_out = (h - size) // stride + 1
    w_out = (w - size) // stride + 1
    out = np.zeros((n, h_out, w_out, c))
    for i in range(h_out):
        for j in range(w_out):
            h_start, h_end = i*stride, i*stride+size
            w_start, w_end = j*stride, j*stride+size
            out[:, i, j, :] = np.max(x[:, h_start:h_end, w_start:w_end, :], axis=(1,2))
    return out

def conv2d(x, w, b, stride=1, padding=1):
    n, h, w_in, c_in = x.shape
    f, f, _, c_out = w.shape
    
    x_padded = np.pad(x, ((0,0),(padding,padding),(padding,padding),(0,0)), mode="constant")
    h_out = (h + 2*padding - f)//stride + 1
    w_out = (w_in + 2*padding - f)//stride + 1
    
    out = np.zeros((n, h_out, w_out, c_out))
    
    for i in range(h_out):
        for j in range(w_out):
            region = x_padded[:, i*stride:i*stride+f, j*stride:j*stride+f, :]
            for k in range(c_out):
                out[:, i, j, k] = np.sum(region * w[:,:,:,k], axis=(1,2,3)) + b[k]
    return out

def flatten(x):
    return x.reshape(x.shape[0], -1)

def fc(x, w, b):
    return x.dot(w) + b


# backwards pass functions
def relu_backward(dout, x):
    """Backward pass for ReLU"""
    return dout * (x > 0)

def fc_backward(dout, x, w, b):
    """Backward pass for fully connected layer"""
    dx = dout.dot(w.T)
    dw = x.T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db

def conv2d_backward(dout, x, w, b, stride=1, padding=1):
    """Backward pass for 2D convolution"""
    n, h, w_in, c_in = x.shape
    f, f, _, c_out = w.shape
    _, h_out, w_out, _ = dout.shape
    
    x_padded = np.pad(x, ((0,0),(padding,padding),(padding,padding),(0,0)), mode="constant")
    
    dx_padded = np.zeros_like(x_padded)
    dw = np.zeros_like(w)
    db = np.sum(dout, axis=(0,1,2))
    
    for i in range(h_out):
        for j in range(w_out):
            h_start, h_end = i*stride, i*stride+f
            w_start, w_end = j*stride, j*stride+f
            
            for k in range(c_out):
                # Gradient w.r.t input - fix broadcasting by reshaping dout properly
                dx_padded[:, h_start:h_end, w_start:w_end, :] += dout[:, i, j, k].reshape(-1,1,1,1) * w[:,:,:,k]
                # Gradient w.r.t weights
                dw[:,:,:,k] += np.sum(x_padded[:, h_start:h_end, w_start:w_end, :] * dout[:, i, j, k].reshape(-1,1,1,1), axis=0)
    
    if padding > 0:
        dx = dx_padded[:, padding:-padding, padding:-padding, :]
    else:
        dx = dx_padded
    
    return dx, dw, db

def max_pool_backward(dout, x, size=2, stride=2):
    """Backward pass for max pooling"""
    n, h, w, c = x.shape
    h_out, w_out = dout.shape[1], dout.shape[2]
    
    dx = np.zeros_like(x)
    
    for i in range(h_out):
        for j in range(w_out):
            h_start, h_end = i*stride, i*stride+size
            w_start, w_end = j*stride, j*stride+size
            
            region = x[:, h_start:h_end, w_start:w_end, :]
            for n_idx in range(n):
                for c_idx in range(c):
                    mask = (region[n_idx, :, :, c_idx] == np.max(region[n_idx, :, :, c_idx]))
                    dx[n_idx, h_start:h_end, w_start:w_end, c_idx] += dout[n_idx, i, j, c_idx] * mask
    
    return dx

def cross_entropy_loss(predictions, targets):
    """Cross entropy loss with softmax"""
    n = predictions.shape[0]
    epsilon = 1e-15
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    loss = -np.sum(targets * np.log(predictions)) / n
    return loss

def cross_entropy_backward(predictions, targets):
    """Backward pass for cross entropy loss"""
    n = predictions.shape[0]
    grad = (predictions - targets) / n
    return grad


# data loading functions
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Load and preprocess a single image"""
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def load_dataset(data_dir):
    """Load dataset from directory structure"""
    classes = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    images = []
    labels = []
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found")
            continue
            
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_dir, img_file)
                img = load_and_preprocess_image(img_path)
                if img is not None:
                    images.append(img)
                    label = np.zeros(len(classes))
                    label[class_to_idx[class_name]] = 1
                    labels.append(label)
    
    return np.array(images), np.array(labels), classes

def create_batches(images, labels, batch_size=8):
    """Create mini-batches from dataset"""
    n_samples = len(images)
    indices = np.random.permutation(n_samples)
    
    batches_x = []
    batches_y = []
    
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        batches_x.append(images[batch_indices])
        batches_y.append(labels[batch_indices])
    
    return batches_x, batches_y


class VGG16:
    def __init__(self, num_classes=1000):
        np.random.seed(42)
        
        self.params = {}
        
        # Block 1: 64 filters
        self.params["conv1_1_w"] = np.random.randn(3,3,3,64) * 0.01
        self.params["conv1_1_b"] = np.zeros(64)
        self.params["conv1_2_w"] = np.random.randn(3,3,64,64) * 0.01
        self.params["conv1_2_b"] = np.zeros(64)
        
        # Block 2: 128 filters
        self.params["conv2_1_w"] = np.random.randn(3,3,64,128) * 0.01
        self.params["conv2_1_b"] = np.zeros(128)
        self.params["conv2_2_w"] = np.random.randn(3,3,128,128) * 0.01
        self.params["conv2_2_b"] = np.zeros(128)
        
        # Block 3: 256 filters
        self.params["conv3_1_w"] = np.random.randn(3,3,128,256) * 0.01
        self.params["conv3_1_b"] = np.zeros(256)
        self.params["conv3_2_w"] = np.random.randn(3,3,256,256) * 0.01
        self.params["conv3_2_b"] = np.zeros(256)
        self.params["conv3_3_w"] = np.random.randn(3,3,256,256) * 0.01
        self.params["conv3_3_b"] = np.zeros(256)
        
        # Block 4: 512 filters
        self.params["conv4_1_w"] = np.random.randn(3,3,256,512) * 0.01
        self.params["conv4_1_b"] = np.zeros(512)
        self.params["conv4_2_w"] = np.random.randn(3,3,512,512) * 0.01
        self.params["conv4_2_b"] = np.zeros(512)
        self.params["conv4_3_w"] = np.random.randn(3,3,512,512) * 0.01
        self.params["conv4_3_b"] = np.zeros(512)
        
        # Block 5: 512 filters
        self.params["conv5_1_w"] = np.random.randn(3,3,512,512) * 0.01
        self.params["conv5_1_b"] = np.zeros(512)
        self.params["conv5_2_w"] = np.random.randn(3,3,512,512) * 0.01
        self.params["conv5_2_b"] = np.zeros(512)
        self.params["conv5_3_w"] = np.random.randn(3,3,512,512) * 0.01
        self.params["conv5_3_b"] = np.zeros(512)
        
        # FC layers
        self.params["fc1_w"] = np.random.randn(512*7*7, 4096) * 0.01
        self.params["fc1_b"] = np.zeros(4096)
        self.params["fc2_w"] = np.random.randn(4096, 4096) * 0.01
        self.params["fc2_b"] = np.zeros(4096)
        self.params["fc3_w"] = np.random.randn(4096, num_classes) * 0.01
        self.params["fc3_b"] = np.zeros(num_classes)
    
    def forward(self, x):
        # Conv Block 1 (64 filters)
        x = relu(conv2d(x, self.params["conv1_1_w"], self.params["conv1_1_b"]))
        x = relu(conv2d(x, self.params["conv1_2_w"], self.params["conv1_2_b"]))
        x = max_pool(x)  # 224x224 -> 112x112
        
        # Conv Block 2 (128 filters)
        x = relu(conv2d(x, self.params["conv2_1_w"], self.params["conv2_1_b"]))
        x = relu(conv2d(x, self.params["conv2_2_w"], self.params["conv2_2_b"]))
        x = max_pool(x)  # 112x112 -> 56x56
        
        # Conv Block 3 (256 filters)
        x = relu(conv2d(x, self.params["conv3_1_w"], self.params["conv3_1_b"]))
        x = relu(conv2d(x, self.params["conv3_2_w"], self.params["conv3_2_b"]))
        x = relu(conv2d(x, self.params["conv3_3_w"], self.params["conv3_3_b"]))
        x = max_pool(x)  # 56x56 -> 28x28
        
        # Conv Block 4 (512 filters)
        x = relu(conv2d(x, self.params["conv4_1_w"], self.params["conv4_1_b"]))
        x = relu(conv2d(x, self.params["conv4_2_w"], self.params["conv4_2_b"]))
        x = relu(conv2d(x, self.params["conv4_3_w"], self.params["conv4_3_b"]))
        x = max_pool(x)  # 28x28 -> 14x14
        
        # Conv Block 5 (512 filters)
        x = relu(conv2d(x, self.params["conv5_1_w"], self.params["conv5_1_b"]))
        x = relu(conv2d(x, self.params["conv5_2_w"], self.params["conv5_2_b"]))
        x = relu(conv2d(x, self.params["conv5_3_w"], self.params["conv5_3_b"]))
        x = max_pool(x)  # 14x14 -> 7x7
        
        # Fully Connected layers
        x = flatten(x)  # Should be (batch_size, 7*7*512)
        x = relu(fc(x, self.params["fc1_w"], self.params["fc1_b"]))
        x = relu(fc(x, self.params["fc2_w"], self.params["fc2_b"]))
        x = fc(x, self.params["fc3_w"], self.params["fc3_b"])
        out = softmax(x)
        return out
    
    def forward_with_cache(self, x):
        """Forward pass that caches intermediate values for backpropagation"""
        cache = {}
        
        # Conv Block 1 (64 filters)
        cache['conv1_1_in'] = x
        x = conv2d(x, self.params["conv1_1_w"], self.params["conv1_1_b"])
        cache['conv1_1_out'] = x
        x = relu(x)
        cache['relu1_1_in'] = cache['conv1_1_out']
        
        cache['conv1_2_in'] = x
        x = conv2d(x, self.params["conv1_2_w"], self.params["conv1_2_b"])
        cache['conv1_2_out'] = x
        x = relu(x)
        cache['relu1_2_in'] = cache['conv1_2_out']
        
        cache['pool1_in'] = x
        x = max_pool(x)  # 224x224 -> 112x112
        
        # Conv Block 2 (128 filters)
        cache['conv2_1_in'] = x
        x = conv2d(x, self.params["conv2_1_w"], self.params["conv2_1_b"])
        cache['conv2_1_out'] = x
        x = relu(x)
        cache['relu2_1_in'] = cache['conv2_1_out']
        
        cache['conv2_2_in'] = x
        x = conv2d(x, self.params["conv2_2_w"], self.params["conv2_2_b"])
        cache['conv2_2_out'] = x
        x = relu(x)
        cache['relu2_2_in'] = cache['conv2_2_out']
        
        cache['pool2_in'] = x
        x = max_pool(x)  # 112x112 -> 56x56
        
        # Conv Block 3 (256 filters)
        cache['conv3_1_in'] = x
        x = conv2d(x, self.params["conv3_1_w"], self.params["conv3_1_b"])
        cache['conv3_1_out'] = x
        x = relu(x)
        cache['relu3_1_in'] = cache['conv3_1_out']
        
        cache['conv3_2_in'] = x
        x = conv2d(x, self.params["conv3_2_w"], self.params["conv3_2_b"])
        cache['conv3_2_out'] = x
        x = relu(x)
        cache['relu3_2_in'] = cache['conv3_2_out']
        
        cache['conv3_3_in'] = x
        x = conv2d(x, self.params["conv3_3_w"], self.params["conv3_3_b"])
        cache['conv3_3_out'] = x
        x = relu(x)
        cache['relu3_3_in'] = cache['conv3_3_out']
        
        cache['pool3_in'] = x
        x = max_pool(x)  # 56x56 -> 28x28
        
        # Conv Block 4 (512 filters)
        cache['conv4_1_in'] = x
        x = conv2d(x, self.params["conv4_1_w"], self.params["conv4_1_b"])
        cache['conv4_1_out'] = x
        x = relu(x)
        cache['relu4_1_in'] = cache['conv4_1_out']
        
        cache['conv4_2_in'] = x
        x = conv2d(x, self.params["conv4_2_w"], self.params["conv4_2_b"])
        cache['conv4_2_out'] = x
        x = relu(x)
        cache['relu4_2_in'] = cache['conv4_2_out']
        
        cache['conv4_3_in'] = x
        x = conv2d(x, self.params["conv4_3_w"], self.params["conv4_3_b"])
        cache['conv4_3_out'] = x
        x = relu(x)
        cache['relu4_3_in'] = cache['conv4_3_out']
        
        cache['pool4_in'] = x
        x = max_pool(x)  # 28x28 -> 14x14
        
        # Conv Block 5 (512 filters)
        cache['conv5_1_in'] = x
        x = conv2d(x, self.params["conv5_1_w"], self.params["conv5_1_b"])
        cache['conv5_1_out'] = x
        x = relu(x)
        cache['relu5_1_in'] = cache['conv5_1_out']
        
        cache['conv5_2_in'] = x
        x = conv2d(x, self.params["conv5_2_w"], self.params["conv5_2_b"])
        cache['conv5_2_out'] = x
        x = relu(x)
        cache['relu5_2_in'] = cache['conv5_2_out']
        
        cache['conv5_3_in'] = x
        x = conv2d(x, self.params["conv5_3_w"], self.params["conv5_3_b"])
        cache['conv5_3_out'] = x
        x = relu(x)
        cache['relu5_3_in'] = cache['conv5_3_out']
        
        cache['pool5_in'] = x
        x = max_pool(x)  # 14x14 -> 7x7
        cache['pool5_out'] = x  # Store the output shape for reshaping
        
        # Fully Connected layers
        cache['fc_in'] = flatten(x)  # Should be (batch_size, 7*7*512)
        x = fc(cache['fc_in'], self.params["fc1_w"], self.params["fc1_b"])
        cache['fc1_out'] = x
        x = relu(x)
        cache['relu_fc1_in'] = cache['fc1_out']
        
        cache['fc2_in'] = x
        x = fc(x, self.params["fc2_w"], self.params["fc2_b"])
        cache['fc2_out'] = x
        x = relu(x)
        cache['relu_fc2_in'] = cache['fc2_out']
        
        cache['fc3_in'] = x
        x = fc(x, self.params["fc3_w"], self.params["fc3_b"])
        out = softmax(x)
        
        return out, cache
    
    def backward(self, dout, cache):
        """Backward pass through the network"""
        grads = {}
        
        # Softmax gradient (assuming cross-entropy loss)
        dx = dout
        
        # FC3 backward
        dx, grads['fc3_w'], grads['fc3_b'] = fc_backward(dx, cache['fc3_in'], self.params['fc3_w'], self.params['fc3_b'])
        
        # ReLU FC2 backward
        dx = relu_backward(dx, cache['relu_fc2_in'])
        
        # FC2 backward
        dx, grads['fc2_w'], grads['fc2_b'] = fc_backward(dx, cache['fc2_in'], self.params['fc2_w'], self.params['fc2_b'])
        
        # ReLU FC1 backward
        dx = relu_backward(dx, cache['relu_fc1_in'])
        
        # FC1 backward
        dx, grads['fc1_w'], grads['fc1_b'] = fc_backward(dx, cache['fc_in'], self.params['fc1_w'], self.params['fc1_b'])
        
        # Reshape for conv layers (back to the shape after pool5)
        dx = dx.reshape(cache['pool5_out'].shape)
        
        # Pool5 backward
        dx = max_pool_backward(dx, cache['pool5_in'])
        
        # Block 5 backward
        dx = relu_backward(dx, cache['relu5_3_in'])
        dx, grads['conv5_3_w'], grads['conv5_3_b'] = conv2d_backward(dx, cache['conv5_3_in'], self.params['conv5_3_w'], self.params['conv5_3_b'])
        
        dx = relu_backward(dx, cache['relu5_2_in'])
        dx, grads['conv5_2_w'], grads['conv5_2_b'] = conv2d_backward(dx, cache['conv5_2_in'], self.params['conv5_2_w'], self.params['conv5_2_b'])
        
        dx = relu_backward(dx, cache['relu5_1_in'])
        dx, grads['conv5_1_w'], grads['conv5_1_b'] = conv2d_backward(dx, cache['conv5_1_in'], self.params['conv5_1_w'], self.params['conv5_1_b'])
        
        # Pool4 backward
        dx = max_pool_backward(dx, cache['pool4_in'])
        
        # Block 4 backward
        dx = relu_backward(dx, cache['relu4_3_in'])
        dx, grads['conv4_3_w'], grads['conv4_3_b'] = conv2d_backward(dx, cache['conv4_3_in'], self.params['conv4_3_w'], self.params['conv4_3_b'])
        
        dx = relu_backward(dx, cache['relu4_2_in'])
        dx, grads['conv4_2_w'], grads['conv4_2_b'] = conv2d_backward(dx, cache['conv4_2_in'], self.params['conv4_2_w'], self.params['conv4_2_b'])
        
        dx = relu_backward(dx, cache['relu4_1_in'])
        dx, grads['conv4_1_w'], grads['conv4_1_b'] = conv2d_backward(dx, cache['conv4_1_in'], self.params['conv4_1_w'], self.params['conv4_1_b'])
        
        # Pool3 backward
        dx = max_pool_backward(dx, cache['pool3_in'])
        
        # Block 3 backward
        dx = relu_backward(dx, cache['relu3_3_in'])
        dx, grads['conv3_3_w'], grads['conv3_3_b'] = conv2d_backward(dx, cache['conv3_3_in'], self.params['conv3_3_w'], self.params['conv3_3_b'])
        
        dx = relu_backward(dx, cache['relu3_2_in'])
        dx, grads['conv3_2_w'], grads['conv3_2_b'] = conv2d_backward(dx, cache['conv3_2_in'], self.params['conv3_2_w'], self.params['conv3_2_b'])
        
        dx = relu_backward(dx, cache['relu3_1_in'])
        dx, grads['conv3_1_w'], grads['conv3_1_b'] = conv2d_backward(dx, cache['conv3_1_in'], self.params['conv3_1_w'], self.params['conv3_1_b'])
        
        # Pool2 backward
        dx = max_pool_backward(dx, cache['pool2_in'])
        
        # Block 2 backward
        dx = relu_backward(dx, cache['relu2_2_in'])
        dx, grads['conv2_2_w'], grads['conv2_2_b'] = conv2d_backward(dx, cache['conv2_2_in'], self.params['conv2_2_w'], self.params['conv2_2_b'])
        
        dx = relu_backward(dx, cache['relu2_1_in'])
        dx, grads['conv2_1_w'], grads['conv2_1_b'] = conv2d_backward(dx, cache['conv2_1_in'], self.params['conv2_1_w'], self.params['conv2_1_b'])
        
        # Pool1 backward
        dx = max_pool_backward(dx, cache['pool1_in'])
        
        # Block 1 backward
        dx = relu_backward(dx, cache['relu1_2_in'])
        dx, grads['conv1_2_w'], grads['conv1_2_b'] = conv2d_backward(dx, cache['conv1_2_in'], self.params['conv1_2_w'], self.params['conv1_2_b'])
        
        dx = relu_backward(dx, cache['relu1_1_in'])
        dx, grads['conv1_1_w'], grads['conv1_1_b'] = conv2d_backward(dx, cache['conv1_1_in'], self.params['conv1_1_w'], self.params['conv1_1_b'])
        
        return grads
    
    def update_parameters(self, grads, learning_rate=0.001):
        """Update parameters using SGD"""
        for key in self.params:
            if key in grads:
                self.params[key] -= learning_rate * grads[key]
    
    def save_model(self, filepath):
        """Save model parameters"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.params, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model parameters"""
        with open(filepath, 'rb') as f:
            self.params = pickle.load(f)
        print(f"Model loaded from {filepath}")
    
    def predict(self, x):
        """Make predictions on input"""
        return self.forward(x)
    
    def evaluate(self, images, labels):
        """Evaluate model accuracy"""
        correct = 0
        total = 0
        
        for i in range(len(images)):
            pred = self.predict(images[i:i+1])
            predicted_class = np.argmax(pred[0])
            true_class = np.argmax(labels[i])
            
            if predicted_class == true_class:
                correct += 1
            total += 1
        
        accuracy = correct / total
        return accuracy

def train_model():
    """Complete training pipeline"""
    print("Loading training data...")
    train_images, train_labels, classes = load_dataset("data_train_normal")
    print(f"Loaded {len(train_images)} training images")
    
    print("Loading test data...")
    test_images, test_labels, _ = load_dataset("data_test")
    print(f"Loaded {len(test_images)} test images")
    
    # Initialize model
    model = VGG16(num_classes=5)
    
    # Training parameters
    learning_rate = 0.0001  # Small learning rate for stability
    batch_size = 4  # Small batch size due to computational complexity
    epochs = 5  # Few epochs for demonstration
    
    print("Starting training...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Create batches
        train_batches_x, train_batches_y = create_batches(train_images, train_labels, batch_size)
        
        total_loss = 0
        num_batches = len(train_batches_x)
        
        for batch_idx, (batch_x, batch_y) in enumerate(zip(train_batches_x, train_batches_y)):
            # Forward pass with cache
            predictions, cache = model.forward_with_cache(batch_x)
            
            # Compute loss
            loss = cross_entropy_loss(predictions, batch_y)
            total_loss += loss
            
            # Backward pass
            dout = cross_entropy_backward(predictions, batch_y)
            grads = model.backward(dout, cache)
            
            # Update parameters
            model.update_parameters(grads, learning_rate)
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{num_batches}, Loss: {loss:.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"Average Loss: {avg_loss:.4f}")
        
        # Evaluate on test set (small sample for speed)
        test_sample_size = min(20, len(test_images))
        test_accuracy = model.evaluate(test_images[:test_sample_size], test_labels[:test_sample_size])
        print(f"Test Accuracy (sample): {test_accuracy:.4f}")
        
        # Save model after each epoch
        model.save_model(f"vgg16_flower_epoch_{epoch+1}.pkl")
    
    print("\nTraining completed!")
    
    # Final evaluation
    final_accuracy = model.evaluate(test_images, test_labels)
    print(f"Final Test Accuracy: {final_accuracy:.4f}")
    
    return model

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        print("Training progress...")
        model = train_model()
    else:
        # test prediction, random weights & random input
        print("Demo mode - testing with fake input...")
        x = np.random.randn(1, 224, 224, 3)
        
        model = VGG16(num_classes=5)
        y = model.forward(x)
        print("Output shape:", y.shape)
        print("Predicted probs:", y)
        
        print("\nfor training, run:")
        print("python VGG.py train")
