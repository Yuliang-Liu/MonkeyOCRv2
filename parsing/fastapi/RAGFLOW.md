# RAGFlow integration

MonkeyOCRv2 exposes a parser endpoint at `POST /file_parse` with the same
wire contract used by MinerU. Configure RAGFlow's external
parser/MinerU URL to the service base URL, for example
`http://monkeyocr:7861`.

The request is `multipart/form-data` with one or more parts named `files`.
Older RAGFlow versions that send a single part named `file` are supported too.
Only PDF and image files are accepted. A successful response contains both the
current top-level form and the wrapped form used by older MinerU releases:

```json
{
  "code": 0,
  "success": true,
  "results": {"report.pdf": {"md_content": "# ...", "images": {}}},
  "data": {"results": {"report.pdf": {"md_content": "# ...", "images": {}}}}
}
```

Start the API after starting the OpenAI-compatible vLLM backend:

```bash
python fastapi/main.py --server-url http://127.0.0.1:8888 --api-port 7861
```

The existing `POST /parse` endpoint remains unchanged and continues to return
a ZIP artifact for direct clients. Set `MOCR2_OUTPUT_DIR` to a persistent
volume when running the service in a container.
