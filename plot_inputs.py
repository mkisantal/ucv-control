import numpy as np
import matplotlib.pyplot as plt


class InputPlotter:
    def __init__(self, client):
        self.clt = client
        return

    @staticmethod
    def read_png(res):
        import StringIO
        import PIL.Image
        img = PIL.Image.open(StringIO.StringIO(res))
        return np.asarray(img)

    @staticmethod
    def read_npy(res):
        import StringIO
        return np.load(StringIO.StringIO(res))

    def _render_frame(self):
        f = dict(
            lit=self.clt.request('vget /camera/0/lit png'),
            depth=self.clt.request('vget /camera/0/depth npy'),
            object_mask=self.clt.request('vget /camera/0/object_mask png'),
            normal=self.clt.request('vget /camera/0/normal png'),
        )
        return f

    def _subplot_image(self, sub_index, image, imtype, param=None):
        if imtype in {'lit', 'object_mask', 'normal'}:
            image = self.read_png(image)
        elif imtype == 'depth':
            image = self.read_npy(image)
        else:
            return
        plt.subplot(sub_index)
        plt.imshow(image, param)
        plt.axis('off')

    def render(self):
        frame = self._render_frame()

        self._subplot_image(221, frame['lit'], 'lit')
        self._subplot_image(222, frame['depth'], 'depth')
        self._subplot_image(223, frame['object_mask'], 'object_mask')
        self._subplot_image(224, frame['normal'], 'normal')

        plt.show()
