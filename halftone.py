# Copyright (c) 2023 Jonathan S. Pollack (https://github.com/JPPhoto)
# Halftoning implementation via Bohumir Zamecnik (https://github.com/bzamecnik/halftone/)

import random
from typing import Callable, Optional, Tuple

import numpy as np
from PIL import Image
from pydantic import BaseModel

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    BaseInvocationOutput,
    InputField,
    InvocationContext,
    OutputField,
    invocation,
)
from invokeai.app.invocations.primitives import ImageField, ImageOutput
from invokeai.app.models.image import ImageCategory, ResourceOrigin


class HalftoneBase:
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

    def euclid_dot(self, spacing: float, angle: float) -> Callable[[int, int], float]:
        pixel_div = 2.0 / spacing

        def func(x: int, y: int):
            x, y = self.rotate(x * pixel_div, y * pixel_div, angle)
            return 0.5 - (0.25 * (np.sin(np.pi * (x + 0.5)) + np.cos(np.pi * y)))

        return func


@invocation("halftone", title="Halftone", tags=["halftone"], version="1.0.0")
class HalftoneInvocation(BaseInvocation, HalftoneBase):
    """Halftones an image"""

    image: ImageField = InputField(description="The image to halftone", default=None)
    spacing: float = InputField(gt=0, le=800, description="Halftone dot spacing", default=8)
    angle: float = InputField(ge=0, lt=360, description="Halftone angle", default=45)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)
        mode = image.mode

        image = image.convert("L")
        image = self.array_from_pil(image)
        halftoned = image > self.evaluate_2d_func(image.shape, self.euclid_dot(self.spacing, self.angle))
        halftoned = self.pil_from_array(halftoned)

        if mode == "RGBA":
            image = halftoned.convert("RGBA")
        else:
            image = halftoned.convert("RGB")

        image_dto = context.services.images.create(
            image=image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=None,
            workflow=self.workflow,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image.width,
            height=image.height,
        )


@invocation("cmyk_halftone", title="CMYK Halftone", tags=["halftone"], version="1.0.0")
class CMYKHalftoneInvocation(BaseInvocation, HalftoneBase):
    """Halftones an image in the style of a CMYK print"""

    image: ImageField = InputField(description="The image to halftone", default=None)
    spacing: float = InputField(gt=0, le=800, description="Halftone dot spacing", default=8)
    c_angle: float = InputField(ge=0, lt=360, description="C halftone angle", default=15)
    m_angle: float = InputField(ge=0, lt=360, description="M halftone angle", default=75)
    y_angle: float = InputField(ge=0, lt=360, description="Y halftone angle", default=90)
    k_angle: float = InputField(ge=0, lt=360, description="K halftone angle", default=45)

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)
        mode = image.mode

        image = image.convert("CMYK")
        c, m, y, k = image.split()

        c = self.array_from_pil(c)
        c = c > self.evaluate_2d_func(c.shape, self.euclid_dot(self.spacing, self.c_angle))
        c = self.pil_from_array(c)

        m = self.array_from_pil(m)
        m = m > self.evaluate_2d_func(m.shape, self.euclid_dot(self.spacing, self.m_angle))
        m = self.pil_from_array(m)

        y = self.array_from_pil(y)
        y = y > self.evaluate_2d_func(y.shape, self.euclid_dot(self.spacing, self.y_angle))
        y = self.pil_from_array(y)

        k = self.array_from_pil(k)
        k = k > self.evaluate_2d_func(k.shape, self.euclid_dot(self.spacing, self.k_angle))
        k = self.pil_from_array(k)

        halftoned = Image.merge("CMYK", (c, m, y, k))
        if mode == "RGBA":
            image = halftoned.convert("RGBA")
        else:
            image = halftoned.convert("RGB")

        image_dto = context.services.images.create(
            image=image,
            image_origin=ResourceOrigin.INTERNAL,
            image_category=ImageCategory.GENERAL,
            node_id=self.id,
            session_id=context.graph_execution_state_id,
            is_intermediate=self.is_intermediate,
            metadata=None,
            workflow=self.workflow,
        )

        return ImageOutput(
            image=ImageField(image_name=image_dto.image_name),
            width=image.width,
            height=image.height,
        )
