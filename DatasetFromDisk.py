import cv2
import tensorflow as tf
import skimage


class DatasetFromDisk:

    def __init__(self, width, height, channels, colormap=cv2.COLORMAP_VIRIDIS):
        self.width = width
        self.height = height
        self.channels = channels
        self.colormap = colormap


    def read_dataset_image(self, path, label):
        # Read the image file with opencv
        image_np = cv2.imread(path.decode("ascii"), cv2.IMREAD_UNCHANGED)
        # Transform image from gray to colored
        image_np = cv2.applyColorMap(cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
                                     self.colormap)
        # Change from BGR to RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        # Resize the image
        image_np = skimage.transform.resize(image_np, (self.height, self.width, self.channels),
                                            preserve_range=True, anti_aliasing=True)

        return tf.convert_to_tensor(image_np, dtype=tf.int32), label

    def process_path(self, path, label):
        # Get the image from path
        image = tf.numpy_function(self.read_dataset_image, [path, label], (tf.int32, tf.int32))[0]
        # Set the shape manually
        image.set_shape([self.height, self.width, self.channels])

        return image, label


    def process_image(self, image):
        processed_image = cv2.applyColorMap(
            cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U), self.colormap)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        processed_image = skimage.transform.resize(processed_image, (self.height, self.width, self.channels),
                                                    preserve_range=True, anti_aliasing=True)
        processed_image = cv2.normalize(processed_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


        return processed_image

    def apply_mask(self, image, mask):
        norm_mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        rescaled_mask = skimage.transform.resize(norm_mask, (self.height, self.width),
                                                 preserve_range=True, anti_aliasing=True)
        norm_mask2 = cv2.normalize(rescaled_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        processed_mask = cv2.bitwise_and(image, image, mask=norm_mask2)

        return processed_mask

    def process_images_and_masks(self, image_paths, masks_paths):
        # Read images and save in disk
        images = [cv2.imread(image_paths[i], cv2.IMREAD_UNCHANGED) for i in range(len(image_paths))]
        masks = [cv2.imread(masks_paths[i], cv2.IMREAD_UNCHANGED) for i in range(len(masks_paths))]

        processed_images = []
        processed_masks = []

        for i in range(len(image_paths)):
            # Process original image
            processed_image = self.process_image(images[i])
            processed_images.append(processed_image)

            # Apply mask to the image
            processed_mask = self.apply_mask(processed_image, masks[i])
            processed_masks.append(processed_mask)

        return processed_images, processed_masks