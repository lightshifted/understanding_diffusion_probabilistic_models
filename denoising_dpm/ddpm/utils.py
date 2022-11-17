import imageio
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    Resize,
    ToTensor,
)
from torchvision.transforms.functional import resize, to_pil_image


def show_image(img, title=""):
    img = img.clip(0, 1)
    img = img.cpu().numpy()
    plt.imshow(img.transpose(1, 2, 0))
    plt.title(title)
    plt.show()


def make_video(path, frames):
    writer = imageio.get_writer(path, len(frames) // 20)

    for f in frames:
        f = f.clip(0, 1)
        f = to_pil_image(resize(f, [368, 368]))
        writer.append_data(np.array(f))

    writer.close()


class ImageLoader:
    def __init__(self, dataset_name="huggan/CelebA-faces", split="train", transforms=True):
        self.dataset_name = dataset_name
        self.image_size = 32
        self.split = split
        self.image_size = 32

        self.transform = Compose(
            [
                Resize(self.image_size),
                CenterCrop(self.image_size),
                ToTensor(),
                Lambda(lambda t: (t * 2) - 1),
            ]
        )

    def transforms(self, examples):
        examples["image"] = [self.transform(image) for image in examples["image"]]
        return examples

    def stage_data(self):
        dataset = load_dataset(self.dataset_name, split=self.split)
        transformed_dataset = dataset.with_transform(self.transforms)
        return transformed_dataset


class CfgNode:
    """a lightweight configuration class inspired by yacs"""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """need to have a helper to support nested indentation for pretty printing"""
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [" " * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """return a dict representation of the config"""
        return {k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items()}

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].
        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:
        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:

            keyval = arg.split("=")
            assert len(keyval) == 2, (
                "expecting each override arg to be of form --arg=value, got %s" % arg
            )
            key, val = keyval  # unpack

            # first translate val into a python object
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == "--"
            key = key[2:]  # strip the '--'
            keys = key.split(".")
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)


# ==========================================
