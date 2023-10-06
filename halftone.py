# Copyright (c) 2023 Jonathan S. Pollack (https://github.com/JPPhoto)
# Halftoning implementation via Bohumir Zamecnik (https://github.com/bzamecnik/halftone/)

import os, os.path
import random
from typing import Callable, Optional, Tuple, Literal

import cv2
import numpy as np
import PIL.ImageCms
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
    def pil_from_array(self, arr, resize_to=None):
        arr = (arr * 255.).astype('uint8')
        if not (resize_to is None):
            arr = cv2.resize(arr, dsize=resize_to, interpolation=cv2.INTER_LANCZOS4)
        return Image.fromarray(arr)

    def array_from_pil(self, img, resize_to=None):
        arr = np.array(img, dtype="uint8")
        if not (resize_to is None):
            arr = cv2.resize(arr, dsize=resize_to, interpolation=cv2.INTER_CUBIC)
        return arr.astype('float64') / 255.

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
            return 0.5 - (0.25 * (np.sin(np.pi * (x + 1.5)) + np.cos(np.pi * (y + 1.))))

        return func_offset if offset else func


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


def load_profiles() -> list:
    """Load available ICC profile filenames from color_profiles/ into a dictionary by name"""
    
    path = "color_profiles"
    profiles = {"None": "None"}
    extensions = [".icc", ".icm"]
    
    if os.path.exists(path):
        for icc_filename in os.listdir(path):
            if icc_filename[-4:].lower() in extensions:
                profile = PIL.ImageCms.getOpenProfile(path + '/' + icc_filename).profile
                description = profile.profile_description
                desc_ext = description[-4:].lower() if (description[-4:].lower() in extensions) else None
                manufacturer = profile.manufacturer
                model = profile.model

                if manufacturer is None:
                    manufacturer = profile.header_manufacturer
                if manufacturer is not None:
                    if manufacturer.isascii() and (not (len(manufacturer.strip('\x00')) == 0)):
                        manufacturer = manufacturer.title()
                    else:
                        manufacturer = None

                name = None
                if ((manufacturer is None) and (model is None)) or  \
                   ((not (manufacturer is None)) and (not (model is None)) and (desc_ext is None)):
                    if desc_ext is None:
                        name = description
                    else:
                        name = description[:-4]
                    name = name.replace('_', ' ')
                elif manufacturer is None:
                    name = model.replace('_', ' ') + "(" + icc_filename + ")"
                elif model is None:
                    name = manufacturer + " : " + '.'.join(icc_filename.split('.')[:-1])
                else:
                    name = manufacturer + " : " + model.replace('_', ' ')

                profiles[name] = icc_filename

    return profiles


color_profiles: list = load_profiles()


@invocation("cmyk_halftone", title="CMYK Halftone", tags=["halftone"], version="1.0.1")
class CMYKHalftoneInvocation(BaseInvocation, HalftoneBase):
    """Halftones an image in the style of a CMYK print"""

    image: ImageField = InputField(description="The image to halftone", default=None)
    spacing: float = InputField(gt=0, le=800, description="Halftone dot spacing", default=8)
    c_angle: float = InputField(ge=0, lt=360, description="C halftone angle", default=15)
    m_angle: float = InputField(ge=0, lt=360, description="M halftone angle", default=75)
    y_angle: float = InputField(ge=0, lt=360, description="Y halftone angle", default=90)
    k_angle: float = InputField(ge=0, lt=360, description="K halftone angle", default=45)
    oversampling: float = InputField(ge=1., le=16., description="Oversampling factor", default=1.)
    offset_c: bool = InputField(default=False, description="Offset Cyan halfway between dots")
    offset_m: bool = InputField(default=False, description="Offset Magenta halfway between dots")
    offset_y: bool = InputField(default=False, description="Offset Yellow halfway between dots")
    offset_k: bool = InputField(default=False, description="Offset K halfway between dots")
    profile: Literal[tuple(color_profiles.keys())] = InputField(
        default=list(color_profiles.keys())[0], description="CMYK Color Profile"
    )

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.services.images.get_pil_image(self.image.image_name)
        mode = image.mode
        size = image.size

        alpha_channel = image.getchannel("A") if (mode == "RGBA") else None

        if self.profile == "None":
            image = image.convert("CMYK")
        else:
            image_rgb = image.convert("RGB")
            cms_profile_cmyk = PIL.ImageCms.getOpenProfile("color_profiles/" + color_profiles[self.profile])
            cms_profile_srgb = PIL.ImageCms.createProfile("sRGB")
            cms_xform = PIL.ImageCms.buildTransformFromOpenProfiles(
                cms_profile_srgb, cms_profile_cmyk, "RGB", "CMYK",
                renderingIntent=PIL.ImageCms.Intent.RELATIVE_COLORIMETRIC,
                flags=(PIL.ImageCms.FLAGS["BLACKPOINTCOMPENSATION"] | PIL.ImageCms.FLAGS["HIGHRESPRECALC"]),
            )
            image = PIL.ImageCms.applyTransform(image_rgb, cms_xform)

        if 1. < self.oversampling:
            image = self.pil_from_array(
                self.array_from_pil(
                    image,
                    resize_to=(int(size[0] * self.oversampling), int(size[1] * self.oversampling))
                )
            )

        c, m, y, k = image.split()

        c = self.array_from_pil(c)
        c = c > self.evaluate_2d_func(c.shape, self.euclid_dot(self.spacing, self.c_angle, self.offset_c))
        c = self.pil_from_array(c)

        m = self.array_from_pil(m)
        m = m > self.evaluate_2d_func(m.shape, self.euclid_dot(self.spacing, self.m_angle, self.offset_m))
        m = self.pil_from_array(m)

        y = self.array_from_pil(y)
        y = y > self.evaluate_2d_func(y.shape, self.euclid_dot(self.spacing, self.y_angle, self.offset_y))
        y = self.pil_from_array(y)

        k = self.array_from_pil(k)
        k = k > self.evaluate_2d_func(k.shape, self.euclid_dot(self.spacing, self.k_angle, self.offset_k))
        k = self.pil_from_array(k)

        halftoned = Image.merge("CMYK", (c, m, y, k))

        if 1. < self.oversampling:
            halftoned = self.pil_from_array(self.array_from_pil(halftoned), resize_to=size)

        if self.profile == "None":
            image = halftoned.convert("RGB")
        else:
            cms_xform = PIL.ImageCms.buildTransformFromOpenProfiles(
                cms_profile_cmyk, cms_profile_srgb, "CMYK", "RGB",
                renderingIntent=PIL.ImageCms.Intent.RELATIVE_COLORIMETRIC,
                flags=(PIL.ImageCms.FLAGS["BLACKPOINTCOMPENSATION"] | PIL.ImageCms.FLAGS["HIGHRESPRECALC"]),
            )
            image = PIL.ImageCms.applyTransform(halftoned, cms_xform)

        if alpha_channel is not None:
            image = Image.merge("RGBA", [
                image.getchannel(0),
                image.getchannel(1),
                image.getchannel(2),
                alpha_channel
            ])

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
