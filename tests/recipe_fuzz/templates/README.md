Pinned Hugging Face chat templates are development-only differential oracles.
Production code neither loads nor interprets these templates.

- Inkling: thinkingmachines/Inkling-Small-NVFP4, b6a99534467840620d411e4cd4ad5819b2610d9c, chat_template.jinja.
- Qwen3.5: Qwen/Qwen3.5-35B-A3B, 59d61f3ce65a6d9863b86d2e96597125219dc754, chat_template.jinja (Apache-2.0).
- Qwen3.8: RadixArk/Qwen3.8-27B-NVFP4, 554ebba9b5f1b79dc11246341960360e6ef05ef4, chat_template.jinja.
- Ornith: ornith-ai/Ornith-1.0-35B, 5df2ed3f675c7beaa490328cc70bb573b65fb660, chat_template.jinja (MIT). A modified Qwen3.5 template: past reasoning is always rendered and non-string tool arguments use JSON.
- GLM-5.3-Flash: nvidia/GLM-5.3-Flash-NVFP4, da920bb0b9f4a06727223a349e55468e38352348, chat_template.jinja (MIT; upstream notice retained at crates/resin-recipe/licenses/glm-MIT.txt).

`glm53_images_hf.npz` contains six float-image processor reference cases from
Hugging Face Transformers revision `50e35cc0ca18ec19b80653690de553e6decf74f5`,
`Glm5NextImageProcessor` (Apache-2.0). Cases cover upscaling, padding without
scaling, downscaling, custom patch size, aligned images, and extreme aspect
ratios. Each stores its RGB float32 input, resulting packed pixels, grid, and
config overrides. The embedded metadata records the source revision. Inputs use
`numpy.random.default_rng(37).random((height, width, 3), dtype=numpy.float32)`.
The reference call uses `do_rescale=False`, `input_data_format="channels_last"`,
and `return_tensors="np"`; unspecified processor parameters use upstream defaults.
`tests/test_glm_media.py` checks these outputs without loading Transformers.
