# Demo cases

`cases.json` contains the 25 examples imported from `2.1 Demo Case.md`: 21 image-editing cases and 4 text-to-image cases. Input images retain their order from the document, including the 10-image room-furnishing example.

Images are bundled locally as lossless WebP files. Decoded RGB/RGBA pixels were verified against the downloaded originals. Reference outputs are displayed separately from generated results; the running-shoe reference was generated separately because the source document did not supply one.

Where the document supplies both `raw_prompt` and `positive_prompt`, the example uses the complete `positive_prompt`. Original fields remain in `source_prompts`. Markdown code fences, table escapes, and encoded line breaks are removed from the displayed prompt. These curated prompts load with prompt enhancement disabled so they are sent unchanged.

Titles have Chinese and English variants. Switching the interface language preserves the original language and content of each prompt.

## Generated reference outputs

The running-shoe case and all 11 original Chinese/English text-to-image examples now include outputs generated with `pre-qwen-image-2.1-eval-preprocess-0917`, seed 42. Their prompt enhancement settings match the corresponding example buttons. `generated_references.json` records the 11 text-to-image prompts, output filenames and generation settings; the running-shoe metadata is in `cases.json`. Generated outputs are stored as lossless WebP. No API keys or temporary download URLs are stored here.

The transparent anime bride example uses the supplied Chinese prompt verbatim with enhancement disabled. Its reference is an original RGBA PNG with verified transparent and semi-transparent pixels; generation metadata includes the request ID.
