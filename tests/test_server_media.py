from __future__ import annotations

import numpy as np
import pytest
from test_recipe_media_preparation import Tokenizer, render_media

import resin
from resin.recipe import get_native_recipe


def test_image_preparation_owns_input_pixels():
    image = np.ones((2, 3, 3), dtype=np.float32)
    recipe = get_native_recipe(
        "inkling2", Tokenizer(), media_config={"do_normalize": False}
    )
    media = render_media(recipe, [resin.recipe.Image.from_rgb(image)])
    prepared = media.pixel_values.copy()
    assert media.pixel_values.flags.c_contiguous
    assert not np.shares_memory(image, media.pixel_values)
    image.fill(0)
    assert np.array_equal(media.pixel_values, prepared)


def test_image_preparation_rejects_implicit_dtype_conversion() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    recipe = get_native_recipe("inkling2", Tokenizer())
    with pytest.raises((TypeError, ValueError), match="float32|dtype"):
        render_media(recipe, [resin.recipe.Image(source=image)])
