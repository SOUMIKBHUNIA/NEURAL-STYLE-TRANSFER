import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image

# Function to load and preprocess an image
def load_image(image_path, max_dim=None):
    """
    Load an image from the given path, decode it, and preprocess it for the model.
    - Resizes the image while maintaining aspect ratio if `max_dim` is specified.
    - Normalizes the image to the range [0, 1].
    """
    img = tf.io.read_file(image_path)  # Read the image file
    img = tf.image.decode_image(img, channels=3)  # Decode the image into a tensor
    img = tf.image.convert_image_dtype(img, tf.float32)  # Convert to float32 in range [0, 1]

    if max_dim:  # Resize the image if a maximum dimension is specified
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        scale = max_dim / max(shape)
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
    
    img = img[tf.newaxis, :]  # Add batch dimension (model expects a batch of images)
    return img

# Function to display images
def display_images(content_image, style_image, styled_image):
    """
    Display the content image, style image, and the resulting styled image side by side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Create a subplot with 3 columns
    axes[0].imshow(np.squeeze(content_image))  # Remove batch dimension and display content image
    axes[0].set_title("Content Image")
    axes[0].axis('off')

    axes[1].imshow(np.squeeze(style_image))  # Remove batch dimension and display style image
    axes[1].set_title("Style Image")
    axes[1].axis('off')

    axes[2].imshow(np.squeeze(styled_image))  # Remove batch dimension and display styled image
    axes[2].set_title("Styled Image")
    axes[2].axis('off')
    plt.show()

# Function to compute the Gram matrix for style representation
def gram_matrix(input_tensor):
    """
    Compute the Gram matrix, which captures the correlation between features in a layer.
    This is used to represent the style of an image.
    """
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)  # Compute feature correlations
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)  # Number of locations in the feature map
    return result / num_locations  # Normalize by the number of locations

# Custom model to extract style and content features
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        """
        Initialize the model:
        - Use VGG19 to extract features from specified style and content layers.
        - The model is non-trainable since we're only using it for feature extraction.
        """
        self.vgg = self._vgg_layers(style_layers + content_layers)  # Build the VGG19 model
        self.style_layers = style_layers  # Layers used for style extraction
        self.content_layers = content_layers  # Layers used for content extraction
        self.num_style_layers = len(style_layers)  # Number of style layers
        self.vgg.trainable = False  # Freeze the VGG19 model

    def call(self, inputs):
        """
        Forward pass through the model:
        - Preprocess the input image for VGG19.
        - Extract style and content features.
        """
        inputs = inputs * 255.0  # Scale the input back to [0, 255] for VGG19 preprocessing
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)  # Preprocess for VGG19
        outputs = self.vgg(preprocessed_input)  # Get outputs from the specified layers
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])  # Split into style and content outputs
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]  # Compute Gram matrices for style
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}  # Map content layers to outputs
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}  # Map style layers to outputs
        return {'content': content_dict, 'style': style_dict}  # Return style and content representations

    def _vgg_layers(self, layer_names):
        """
        Build a VGG19 model that outputs intermediate activations from the specified layers.
        """
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')  # Load VGG19 pretrained on ImageNet
        vgg.trainable = False  # Freeze the model
        outputs = [vgg.get_layer(name).output for name in layer_names]  # Get outputs from specified layers
        model = tf.keras.Model([vgg.input], outputs)  # Create a model that outputs these activations
        return model

# Function to compute the total loss (style + content)
def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight):
    """
    Compute the total loss as a weighted sum of style loss and content loss.
    """
    style_outputs = outputs['style']  # Extract style outputs
    content_outputs = outputs['content']  # Extract content outputs

    # Compute style loss (mean squared error between Gram matrices)
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name]) ** 2)
                           for name in style_outputs.keys()])
    style_loss *= style_weight / len(style_layers)  # Normalize by the number of style layers

    # Compute content loss (mean squared error between feature maps)
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name]) ** 2)
                             for name in content_outputs.keys()])
    content_loss *= content_weight / len(content_layers)  # Normalize by the number of content layers

    return style_loss + content_loss  # Total loss is the sum of style and content losses

# Main function for neural style transfer
def neural_style_transfer(content_path, style_path, iterations=1000, style_weight=1e-2, content_weight=1e4):
    """
    Perform Neural Style Transfer:
    - Load content and style images.
    - Extract style and content features using VGG19.
    - Optimize a generated image to match both style and content.
    """
    # Load content and style images
    content_image = load_image(content_path, max_dim=512)  # Load and preprocess content image
    style_image = load_image(style_path, max_dim=512)  # Load and preprocess style image

    # Define layers for VGG19
    content_layers = ['block5_conv2']  # Layers used for content representation
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']  # Layers used for style representation

    # Build the model for extracting style and content features
    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)['style']  # Extract style features from the style image
    content_targets = extractor(content_image)['content']  # Extract content features from the content image

    # Initialize the generated image (start with the content image)
    generated_image = tf.Variable(content_image)

    # Define optimizer
    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    # Training step (optimize the generated image)
    @tf.function()
    def train_step(image):
        """
        Perform one optimization step:
        - Compute gradients of the loss with respect to the generated image.
        - Update the generated image using the optimizer.
        """
        with tf.GradientTape() as tape:
            outputs = extractor(image)  # Extract style and content features from the generated image
            loss = style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight)  # Compute total loss
        grad = tape.gradient(loss, image)  # Compute gradients of the loss with respect to the generated image
        opt.apply_gradients([(grad, image)])  # Update the generated image
        image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))  # Clip values to [0, 1]

    # Run optimization
    for i in range(iterations):  # Perform multiple iterations of optimization
        train_step(generated_image)  # Optimize the generated image
        if i % 100 == 0:  # Print progress every 100 iterations
            print(f"Iteration {i}")

    # Display results
    display_images(content_image, style_image, generated_image)

# Example usage
if __name__ == "__main__":
    content_path = "path_to_your_content_image.jpg"  # Replace with your content image path
    style_path = "path_to_your_style_image.jpg"      # Replace with your style image path
    neural_style_transfer(content_path, style_path)
