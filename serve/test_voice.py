"""local (Indra's Mirror): the voice sidecar - config defaults, health, and the passthrough forward.

The start/stop idle loop is the same shape as the tested lazy vision; the spawn path needs a real
audio server, so it is exercised here only for the clean-failure case.
"""
import http.server
import json
import threading
import unittest

from serve.voice import VoiceServer


class _StubHandler(http.server.BaseHTTPRequestHandler):
    seen = None

    def do_GET(self):
        self._send(200, b"ok", "text/plain") if self.path == "/health" else self._send(404, b"{}", "application/json")

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        _StubHandler.seen = (self.path, self.headers.get("Content-Type"), body)
        self._send(200, json.dumps({"text": "hello world", "bytes": len(body)}).encode(), "application/json")

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class VoiceServerTest(unittest.TestCase):
    def setUp(self):
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _StubHandler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.addCleanup(self.httpd.shutdown)

    def test_config_defaults(self):
        v = VoiceServer("asr", {"cmd": "/bin/true"})
        self.assertEqual(v.port, 5110)
        self.assertEqual(v.reserve_mib, 1800)
        self.assertEqual(v.base_url, "http://127.0.0.1:5110")
        self.assertFalse(v.alive())          # not started
        v.stop()                             # safe when it was never started

    def test_tts_default_port(self):
        self.assertEqual(VoiceServer("tts", {"cmd": "/bin/true"}).port, 5210)

    def test_forward_passes_the_body_through_and_parses_the_reply(self):
        v = VoiceServer("asr", {"cmd": "/bin/true", "port": self.port})
        self.assertTrue(v._healthy())
        status, out, _ctype = v.forward("/v1/audio/transcriptions", b"RIFFxxxx", "audio/wav")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(out)["text"], "hello world")
        self.assertEqual(_StubHandler.seen, ("/v1/audio/transcriptions", "audio/wav", b"RIFFxxxx"))

    def test_ensure_fails_cleanly_when_the_command_is_missing(self):
        v = VoiceServer("asr", {"cmd": "/nonexistent/definitely-not-here", "port": self.port,
                                "start_timeout_s": 2})
        with self.assertRaises(RuntimeError):
            v.ensure()
        self.assertFalse(v.alive())

    def test_ensure_counts_the_start_as_use(self):
        # the idle loop stops a child once time.time() - last_used >= idle_s; before this, last_used stayed 0
        # until the caller touched it AFTER ensure(), so a tick during a slow start killed the first request
        import time
        v = VoiceServer("asr", {"cmd": "/bin/sleep", "args": ["30"], "port": self.port, "start_timeout_s": 5})
        self.addCleanup(v.stop)
        v.ensure()                               # the stub answers /health, so this returns at once
        self.assertTrue(v.alive())
        self.assertLess(time.time() - v.last_used, 5.0)

if __name__ == "__main__":
    unittest.main()
