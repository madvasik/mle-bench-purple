import base64
import io
import logging
import tarfile
from pathlib import Path

from a2a.server.tasks import TaskUpdater
from a2a.types import FilePart, FileWithBytes, Message, Part, TextPart
from a2a.utils import get_message_text

from pipeline import run_tabular_baseline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WORK_ROOT = Path("/home/agent/mle_work")


def _extract_tar_b64(b64_text: str, dest: Path) -> None:
    raw = base64.b64decode(b64_text)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")


def _first_tar_from_message(message: Message) -> str | None:
    for part in message.parts:
        root = part.root
        if isinstance(root, FilePart):
            fd = root.file
            if isinstance(fd, FileWithBytes) and fd.bytes is not None:
                raw = fd.bytes
                if isinstance(raw, str):
                    return raw
                if isinstance(raw, (bytes, bytearray)):
                    return base64.b64encode(raw).decode("ascii")
    return None


class Agent:
    def __init__(self):
        self._done_context: set[str] = set()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        ctx = message.context_id or "default"
        text = get_message_text(message)

        if ctx in self._done_context:
            logger.info("Context %s already finished; ack", ctx)
            return

        tar_b64 = _first_tar_from_message(message)
        if not tar_b64:
            logger.error("No competition tar.gz in message; text len=%s", len(text))
            await updater.add_artifact(
                parts=[Part(root=TextPart(text="Error: expected FilePart competition.tar.gz"))],
                name="Error",
            )
            return

        work_dir = WORK_ROOT / ctx
        if work_dir.exists():
            import shutil

            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        try:
            _extract_tar_b64(tar_b64, work_dir)
        except Exception as e:
            logger.exception("Extract failed")
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=f"Error extracting tar: {e}"))],
                name="Error",
            )
            return

        data_root = work_dir
        inner = work_dir / "home" / "data"
        if inner.is_dir():
            data_root = inner

        try:
            csv_bytes = run_tabular_baseline(data_root)
        except Exception as e:
            logger.exception("Pipeline failed")
            await updater.add_artifact(
                parts=[Part(root=TextPart(text=f"Pipeline error: {e}"))],
                name="Error",
            )
            return

        b64_out = base64.b64encode(csv_bytes).decode("ascii")
        await updater.add_artifact(
            parts=[
                Part(
                    root=FilePart(
                        file=FileWithBytes(
                            bytes=b64_out,
                            name="submission.csv",
                            mime_type="text/csv",
                        )
                    )
                )
            ],
            name="submission",
        )
        self._done_context.add(ctx)
        logger.info("Submitted submission.csv (%s bytes)", len(csv_bytes))
