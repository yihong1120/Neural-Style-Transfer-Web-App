import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import Model
import os


class StyleTransfer:
    def __init__(self, content_path, style_path, max_dim=512, content_weight=1e4, style_weight=1e-2, 
                 content_layers=['block5_conv2'], style_layers=['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']):
        self.content_path = content_path
        self.style_path = style_path
        self.max_dim = max_dim
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.content_layers = content_layers
        self.style_layers = style_layers

    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')  # 將圖像轉換為 RGB 格式
        if self.max_dim:
            factor = self.max_dim / max(image.size)
            size = tuple((round(dim * factor) for dim in image.size))
            image = image.resize(size, Image.ANTIALIAS)
        img_array = np.array(image)
        img_array = tf.expand_dims(img_array, 0)
        return img_array

    def get_model(self, layer_names):
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in layer_names]
        model = Model([vgg.input], outputs)
        return model

    def content_loss(self, base_content, target_content):
        return tf.reduce_mean(tf.square(base_content - target_content))

    def gram_matrix(self, input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations

    def style_loss(self, base_style, target_style):
        base_style_gram = self.gram_matrix(base_style)
        target_style_gram = self.gram_matrix(target_style)
        return tf.reduce_mean(tf.square(base_style_gram - target_style_gram))

    def style_transfer(self, iterations=100):
        content_image = self.load_image(self.content_path)
        style_image = self.load_image(self.style_path)
        extractor = self.get_model(self.content_layers + self.style_layers)
        extracted_outputs = extractor(content_image)
        content_target = [output for output in extracted_outputs[:len(self.content_layers)]]
        extracted_outputs = extractor(style_image)
        style_targets = [output for output in extracted_outputs[len(self.content_layers):]]
        generated_image = tf.Variable(tf.cast(content_image, tf.float32))
        optimizer = tf.optimizers.Adam(learning_rate=0.1, beta_1=0.99, epsilon=1e-1)

        for i in range(iterations):
            with tf.GradientTape() as tape:
                extracted_outputs = extractor(generated_image)
                content_output = [output for output in extracted_outputs[:len(self.content_layers)]]
                style_outputs = [output for output in extracted_outputs[len(self.content_layers):]]
                content_loss_value = self.content_loss(content_output[0], content_target[0])
                style_loss_value = 0
                for target, output in zip(style_targets, style_outputs):
                    style_loss_value += self.style_loss(output, target) / len(self.style_layers)
                total_loss = self.content_weight * content_loss_value + self.style_weight * style_loss_value

            gradients = tape.gradient(total_loss, generated_image)
            optimizer.apply_gradients([(gradients, generated_image)])

            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {total_loss.numpy()}")
                generated_img = generated_image.numpy()
                generated_img = np.squeeze(generated_img, axis=0)
                generated_img = np.clip(generated_img, 0, 255).astype('uint8')
                #plt.imshow(generated_img)
                #plt.show()

        return generated_image

    def save_image(self, image_array, filename):
        image_array = np.squeeze(image_array, axis=0)
        image_array = np.clip(image_array, 0, 255).astype('uint8')
        image = Image.fromarray(image_array)
        image.save(filename)

    def process_folder(self, input_folder, style_path, output_folder=None):
        if output_folder is None:
            output_folder = input_folder

        file_list = os.listdir(input_folder)
        image_files = [file for file in file_list if file.endswith(('.jpg', '.png', '.jpeg', '.webp', '.gif')) and file != os.path.basename(style_path)]

        for image_file in image_files:
            print(f"Processing {image_file}")
            content_path = os.path.join(input_folder, image_file)
            self.content_path = content_path  # 更新實例變量
            output_image = self.style_transfer()
            output_filename = f"transfer_{image_file}"
            output_path = os.path.join(output_folder, output_filename)
            self.save_image(output_image, output_path)

if __name__ == "__main__":
    # input_folder = "static/puppies"
    # style_path = "static/puppies/style_index.jpg"
    # style_transfer = StyleTransfer(None, style_path)  # 先將 content_path 設為 None
    # style_transfer.process_folder('static/puppies', 'static/puppies/style_index.jpg')

    content_path = '17898463157018.jpg'
    style_path = 'Picasso.jpg'
    max_dim = 512 #可使用預設參數
    content_weight = 1e-5# 1e4 #可使用預設參數
    style_weight = 1e3# 1e-2 #可使用預設參數
    content_layers = ['block5_conv2'] #可使用預設參數
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'] #可使用預設參數
    #style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
    # style_layers = ['block1_conv2', 'block2_conv2', 'block3_conv2', 'block4_conv2']
    #style_layers = ['block1_conv1', 'block2_conv2', 'block3_conv3', 'block4_conv4']
    #style_layers = ['block1_conv2', 'block2_conv1', 'block3_conv1', 'block4_conv2', 'block5_conv1']

    iterations = 1000 #可使用預設參數
    output_path = 'output_image.jpg'

    style_transfer = StyleTransfer(content_path, style_path, max_dim, content_weight, style_weight, content_layers, style_layers)
    output_image = style_transfer.style_transfer(iterations=iterations)
    style_transfer.save_image(output_image, output_path)
