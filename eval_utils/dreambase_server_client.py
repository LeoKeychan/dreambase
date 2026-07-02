from __future__ import annotations

import logging
import time
from typing import Any

import websockets.sync.client
from openpi_client import msgpack_numpy


PING_INTERVAL_SECS = 60
PING_TIMEOUT_SECS = 1200


class DreamBasePolicyClient:
    """Thin client for the DreamBase raw-observation policy server."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws, self._metadata = self._connect()

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _connect(self):
        logging.info("Connecting to DreamBase server at %s", self._uri)
        conn = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            ping_interval=PING_INTERVAL_SECS,
            ping_timeout=PING_TIMEOUT_SECS,
        )
        metadata = msgpack_numpy.unpackb(conn.recv())
        return conn, metadata

    def infer(
        self,
        obs: dict[str, Any],
        *,
        return_video_pred: bool = False,
        return_decoded_video: bool = False,
    ) -> dict[str, Any]:
        request = dict(obs)
        request["endpoint"] = "infer"
        request["_return_video_pred"] = bool(return_video_pred)
        request["_return_decoded_video"] = bool(return_decoded_video)
        request["_client_send_time"] = time.time()
        self._ws.send(self._packer.pack(request))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in DreamBase inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def reset(self, reset_info: dict[str, Any] | None = None) -> Any:
        request = dict(reset_info or {})
        request["endpoint"] = "reset"
        self._ws.send(self._packer.pack(request))
        response = self._ws.recv()
        if isinstance(response, bytes):
            return msgpack_numpy.unpackb(response)
        return response
