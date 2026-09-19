"""How much input the service will accept, in one place.

Three modules need to agree on these: the factory configures Werkzeug with
them, the batch route enforces the per-file ceiling, and the error handler
quotes the limit back to the caller.
"""

from __future__ import annotations

#: The largest CSV the batch endpoint will score.
MAX_UPLOAD_BYTES = 5 * 1024 * 1024

#: `MAX_CONTENT_LENGTH` measures the whole multipart body -- part headers and
#: boundaries included -- not the file inside it. Without slack, a CSV of
#: exactly the limit is refused for its envelope. Werkzeug still stops a
#: genuinely huge upload before it is buffered; the route then checks the file
#: itself against `MAX_UPLOAD_BYTES` so the stated limit is the enforced one.
MULTIPART_SLACK_BYTES = 64 * 1024
