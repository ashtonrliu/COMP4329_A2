
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def create_label_to_index_dictionary(cfg):
    """
    A dictionary which transforms labels from [1, 19] to an index used for one-hot encoding
    """
    dictionary = {}
    index = 0

    for label in cfg["labels"]["classes"]:
        dictionary[label] = index
        index += 1
    
    return dictionary

def letterbox_resize(img, cfg, pad_color=(0, 0, 0)) -> Image.Image:
    """
    Resize a PIL image so the longer side equals `target_size`, keeping aspect ratio,
    then pad the shorter side with `pad_color` to make a square image of size target_sizextarget_size.
    
    Args:
        img (PIL.Image): input image
        target_size (int): final square side length (e.g. 320)
        pad_color (tuple/int): RGB or grayscale pad value
    
    Returns:
        PIL.Image: 320x320 letter-boxed image
    """
    target_size = cfg["image"]["resized_size"]
    w, h = img.size
    
    # ---- 1. Compute new size that preserves aspect ratio ---- #
    if w > h:
        new_w = target_size
        new_h = int(round(h * target_size / w))
    else:
        new_h = target_size
        new_w = int(round(w * target_size / h))
    
    # ---- 2. Resize with high-quality interpolation ---- #
    img = img.resize((new_w, new_h), Image.BILINEAR)
    
    # ---- 3. Create padded canvas and paste resized image ---- #
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    
    # Center the image (optional: randomize offsets for extra augmentation)
    left   = pad_w // 2
    top    = pad_h // 2

    if img.mode == "RGB":
        pad_color = pad_color if isinstance(pad_color, tuple) else (pad_color,)*3
    else:
        pad_color = pad_color if isinstance(pad_color, int) else pad_color[0]
    
    new_img = Image.new(img.mode, (target_size, target_size), pad_color)
    new_img.paste(img, (left, top))
    
    return new_img

class ImageRecord:

    def __init__(self, id, image, caption, labels, label_to_index_dict, cfg):
        """
        Collects and stores relevant information about each data entry.
        Attributes:
            - id, retrieved from the filename
            - image, a PIL image best used for displaying
            - label, a string containing labels. If there is more than one, separated by a space
            - captions, a string containing a description of the image
        """
        self.id = id
        # self.image  = image 
        self.image = letterbox_resize(image, cfg) # Reshapes original image to 320x320
        self.caption = caption
        self.label  = labels

        if labels != None:
            self.one_hot_encode = self.create_one_hot_vector(label_to_index_dict, cfg) # Transforms label into one_hot_encoding
    
    def display_data(self):
        plt.imshow(self.image)
        plt.axis('off')      
        plt.title(f"ImageID: {self.id} Label: {self.label}\n{self.caption}", wrap=True)
        plt.show()

    def get_filename(self):
        return f"{self.id}.jpg"
    
    def create_one_hot_vector(self, label_to_index_dict, cfg):
        labels = self.label.split(" ")
        
        one_hot_encode = np.zeros(cfg["labels"]["num_of_labels"])
        for label in labels:
            one_hot_encode[label_to_index_dict.get(label)] = 1  
        
        return one_hot_encode

def retrieve_image(filename, cfg):
    """
    Returns an image from a given filename
    """
    return Image.open(f'{cfg["data"]["basepath"]}/{cfg["data"]["image_directory"]}/{filename}')

def create_all_samples(dataframe, cfg, is_training=True):
    """
    Uses given dataframe to generate all samples which contain an id, image, label and caption.

    Returns a list of samples
    """
    samples = []
    label = None

    label_to_index_dict = create_label_to_index_dictionary(cfg)

    for index, row in dataframe.iterrows():
        filename = row["ImageID"]
        image = retrieve_image(filename, cfg)
        caption = row["Caption"]

        if is_training:
            label = row["Labels"]

        sample = ImageRecord(filename[:-4], image, caption, label, label_to_index_dict, cfg)
        samples.append(sample)
    
    return samples


# test_samples = create_all_samples(test_data, is_training=False)



