# Copyright (c) 2023 Jonathan S. Pollack (https://github.com/JPPhoto)
# Halftoning implementation via Bohumir Zamecnik (https://github.com/bzamecnik/halftone/)

from typing import Callable, Tuple

import numpy as np
from PIL import Image

from invokeai.app.invocations.fields import WithBoard
from invokeai.invocation_api import (
    BaseInvocation,
    ImageField,
    ImageOutput,
    InputField,
    InvocationContext,
    WithMetadata,
    invocation,
)


class HalftoneBase(WithMetadata):
    def pil_from_array(self, arr):
        return Image.fromarray((arr * 255).astype("uint8"))

    def array_from_pil(self, img):
        return np.array(img) / 255

    def evaluate_2d_func(self, img_shape, fn):
        w, h = img_shape
        xaxis, yaxis = np.arange(w), np.arange(h)
        return fn(xaxis[:, None], yaxis[None, :])

    def rotate(self, x: float, y: float, angle: float) -> Tuple[float, float]:
        """
        Rotate coordinates (x, y) by given angle.

        angle: Rotation angle in degrees
        """
        angle_rad = 2 * np.pi * angle / 360
        sin, cos = np.sin(angle_rad), np.cos(angle_rad)
        return x * cos - y * sin, x * sin + y * cos

    def euclid_dot(self, spacing: float, angle: float, offset: bool = False) -> Callable[[int, int], float]:
        pixel_div = 2.0 / spacing

        def func(x: int, y: int):
            x, y = self.rotate(x * pixel_div, y * pixel_div, angle)
            return 0.5 - (0.25 * (np.sin(np.pi * (x + 0.5)) + np.cos(np.pi * y)))

        def func_offset(x: int, y: int):
            x, y = self.rotate(x * pixel_div, y * pixel_div, angle)
            return 0.5 - (0.25 * (np.sin(np.pi * (x + 1.5)) + np.cos(np.pi * (y + 1.0))))

        return func_offset if offset else func


@invocation("halftone", title="Halftone", tags=["halftone"], version="1.1.1")
class HalftoneInvocation(BaseInvocation, HalftoneBase, WithBoard):
    """Halftones an image"""

    image: ImageField = InputField(description="The image to halftone")
    spacing: float = InputField(gt=0, le=800, description="Halftone dot spacing", default=8)
    angle: float = InputField(ge=0, lt=360, description="Halftone angle", default=45)
    oversampling: int = InputField(ge=1, le=4, description="Oversampling factor", default=1)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)
        mode = image.mode
        width, height = image.size

        alpha_channel = image.getchannel("A") if mode == "RGBA" else None

        image = image.convert("L")

        image = image.resize((width * self.oversampling, height * self.oversampling))
        image = self.array_from_pil(image)
        image = image >= self.evaluate_2d_func(
            image.shape, self.euclid_dot(self.spacing * self.oversampling, self.angle, False)
        )
        image = self.pil_from_array(image)
        image = image.resize((width, height))

        image = image.convert("RGB")

        # Make the image RGBA if we had a source alpha channel
        if alpha_channel is not None:
            image.putalpha(alpha_channel)

        image_dto = context.images.save(image=image)

        return ImageOutput.build(image_dto)


@invocation("cmyk_halftone", title="CMYK Halftone", tags=["halftone"], version="1.1.1")
class CMYKHalftoneInvocation(BaseInvocation, HalftoneBase, WithBoard):
    """Halftones an image in the style of a CMYK print"""

    image: ImageField = InputField(description="The image to halftone")
    spacing: float = InputField(gt=0, le=800, description="Halftone dot spacing", default=8)
    c_angle: float = InputField(ge=0, lt=360, description="C halftone angle", default=15)
    m_angle: float = InputField(ge=0, lt=360, description="M halftone angle", default=75)
    y_angle: float = InputField(ge=0, lt=360, description="Y halftone angle", default=90)
    k_angle: float = InputField(ge=0, lt=360, description="K halftone angle", default=45)
    oversampling: int = InputField(ge=1, le=4, description="Oversampling factor", default=1)
    offset_c: bool = InputField(default=False, description="Offset Cyan halfway between dots")
    offset_m: bool = InputField(default=False, description="Offset Magenta halfway between dots")
    offset_y: bool = InputField(default=False, description="Offset Yellow halfway between dots")
    offset_k: bool = InputField(default=False, description="Offset K halfway between dots")

    def convert_rgb_to_cmyk(self, image: Image) -> Image:
        r = self.array_from_pil(image.getchannel("R"))
        g = self.array_from_pil(image.getchannel("G"))
        b = self.array_from_pil(image.getchannel("B"))

        k = 1 - np.maximum(np.maximum(r, g), b)
        c = (1 - r - k) / (1 - k)
        m = (1 - g - k) / (1 - k)
        y = (1 - b - k) / (1 - k)

        c = self.pil_from_array(c)
        m = self.pil_from_array(m)
        y = self.pil_from_array(y)
        k = self.pil_from_array(k)

        return Image.merge("CMYK", (c, m, y, k))

    def convert_cmyk_to_rgb(self, image: Image) -> Image:
        c = self.array_from_pil(image.getchannel("C"))
        m = self.array_from_pil(image.getchannel("M"))
        y = self.array_from_pil(image.getchannel("Y"))
        k = self.array_from_pil(image.getchannel("K"))

        r = (1 - c) * (1 - k)
        g = (1 - m) * (1 - k)
        b = (1 - y) * (1 - k)

        r = self.pil_from_array(r)
        g = self.pil_from_array(g)
        b = self.pil_from_array(b)

        return Image.merge("RGB", (r, g, b))

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)
        mode = image.mode
        width, height = image.size

        alpha_channel = image.getchannel("A") if mode == "RGBA" else None

        image = self.convert_rgb_to_cmyk(image)

        c, m, y, k = image.split()

        c = c.resize((width * self.oversampling, height * self.oversampling))
        c = self.array_from_pil(c)
        c = c >= self.evaluate_2d_func(
            c.shape, self.euclid_dot(self.spacing * self.oversampling, self.c_angle, self.offset_c)
        )
        c = self.pil_from_array(c)
        c = c.resize((width, height))

        m = m.resize((width * self.oversampling, height * self.oversampling))
        m = self.array_from_pil(m)
        m = m >= self.evaluate_2d_func(
            m.shape, self.euclid_dot(self.spacing * self.oversampling, self.m_angle, self.offset_m)
        )
        m = self.pil_from_array(m)
        m = m.resize((width, height))

        y = y.resize((width * self.oversampling, height * self.oversampling))
        y = self.array_from_pil(y)
        y = y >= self.evaluate_2d_func(
            y.shape, self.euclid_dot(self.spacing * self.oversampling, self.y_angle, self.offset_y)
        )
        y = self.pil_from_array(y)
        y = y.resize((width, height))

        k = k.resize((width * self.oversampling, height * self.oversampling))
        k = self.array_from_pil(k)
        k = k >= self.evaluate_2d_func(
            k.shape, self.euclid_dot(self.spacing * self.oversampling, self.k_angle, self.offset_k)
        )
        k = self.pil_from_array(k)
        k = k.resize((width, height))

        image = Image.merge("CMYK", (c, m, y, k))

        image = self.convert_cmyk_to_rgb(image)

        if alpha_channel is not None:
            image.putalpha(alpha_channel)

        image_dto = context.images.save(image=image)

        return ImageOutput.build(image_dto)
