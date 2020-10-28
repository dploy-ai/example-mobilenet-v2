import hashlib
import os

from PIL import ImageColor, ImageFont, Image, ImageDraw


class Rectangle:
    """Rectangle object having 4 coordinates per each corner"""

    def __init__(self, y_min: float, x_min: float, y_max: float, x_max: float) -> None:
        self.y_min = y_min
        self.x_min = x_min
        self.y_max = y_max
        self.x_max = x_max

    def draw(self, pil_draw: ImageDraw, **kwargs) -> None:
        """
        :arg
            pil_draw: PIL Draw object
        """
        pil_draw.rectangle([self.x_min, self.y_min, self.x_max, self.y_max], **kwargs)


class BoundingBox:
    def __init__(self, base_image: Image):
        """
        :arg
            base_image: PIL Image object that will be used as base image and the
                        bounding box will be plotted on this base image
        """
        self.base_image = base_image

    def draw_bounding_box_to_base_image(
        self, rectangle: Rectangle, predicted_class: str, thickness: float
    ) -> ImageDraw:
        """
        :arg
            rectangle: Rectangle object having the coordinates of the bounding box
            thickness: Static integer represents the thickness of the rectangle draw
        """
        scaled_rectangle = self._rescale_rectangle(rectangle)
        pil_draw = ImageDraw.Draw(self.base_image)
        color = self._get_random_color(predicted_class)
        scaled_rectangle.draw(pil_draw, outline=color, width=thickness)
        return pil_draw

    def draw_predicted_class(self, rectangle: Rectangle, display_str: str) -> None:
        scaled_rectangle = self._rescale_rectangle(rectangle)
        pil_draw = ImageDraw.Draw(self.base_image)
        image_size: int = self.base_image.size[0]
        font_size: int = 1  # starting font size

        # portion of image width you want text width to be
        img_fraction = 0.2
        ttf_loc = os.path.join("..", "arial.ttf")
        font = ImageFont.truetype(ttf_loc, font_size)
        while font.getsize(display_str)[0] < img_fraction * image_size:
            # iterate until the text size is just larger than the criteria
            font_size += 1
            font = ImageFont.truetype(ttf_loc, font_size)

        # Find coordinates of bounding text box
        w, h = font.getsize(display_str)
        x = scaled_rectangle.x_min
        y = scaled_rectangle.y_min
        text_rectangle = Rectangle(y, x, y + h, x + w)

        # Find predicted class to get the unique color per class
        predicted_class = display_str.split(":")[0].strip()
        color = self._get_random_color(predicted_class)
        text_rectangle.draw(pil_draw, fill=color)
        pil_draw.text((x, y), display_str, font=font)

    def _rescale_rectangle(self, rectangle: Rectangle) -> Rectangle:
        # Image sizes
        im_width, im_height = self.base_image.size

        scaled_rectangle = Rectangle(
            rectangle.y_min * im_height,
            rectangle.x_min * im_width,
            rectangle.y_max * im_height,
            rectangle.x_max * im_width,
        )
        return scaled_rectangle

    @staticmethod
    def _get_random_color(predicted_class: str) -> str:
        """
        Choose a random color per class
        """
        colors = list(ImageColor.colormap.values())
        color = colors[
            int(hashlib.sha256(predicted_class.encode("utf-8")).hexdigest(), 16)
            % len(colors)
        ]
        return darken(color)


def darken(hex_color: str) -> str:
    """
    Darken an RGB color by 20%
    """
    amount = 0.2
    hex_color = hex_color.replace("#", "")
    red = max(0, int(hex_color[0:2], 16) - int(255 * amount))
    green = max(0, int(hex_color[2:4], 16) - int(255 * amount))
    blue = max(0, int(hex_color[4:6], 16) - int(255 * amount))
    darker_color = (
        "#%s" % hex(red)[2:].zfill(2) + hex(green)[2:].zfill(2) + hex(blue)[2:].zfill(2)
    )
    return darker_color
