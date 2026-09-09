import asyncio
import json
import subprocess
import sys
import textwrap

from sovereign_api.contracts import ModelRequest
from sovereign_api.providers import MockProvider


def test_mock_provider_is_deterministic() -> None:
    provider = MockProvider()
    request = ModelRequest(model_id="mock-code", prompt="Anything")

    first = asyncio.run(provider.generate(request))
    second = asyncio.run(provider.generate(request))

    assert first == second
    assert first.content == "Mock response from mock-code"


def test_mock_provider_has_no_external_network_dependency() -> None:
    # Audit hooks cannot be removed, so isolate this process-wide guard. Installing
    # it before imports also catches socket constructors captured by import alias.
    script = textwrap.dedent(
        """
        import sys

        class NetworkDenied(RuntimeError):
            pass

        denied_events = []
        guarded_events = {
            "socket.connect",
            "socket.getaddrinfo",
            "socket.gethostbyaddr",
            "socket.gethostbyname",
            "socket.sendto",
        }

        def deny_network(event, _args):
            if event in guarded_events:
                denied_events.append(event)
                raise NetworkDenied(event)

        sys.addaudithook(deny_network)

        from socket import AF_UNIX, SOCK_STREAM, socket as captured_socket

        canary_blocked = False
        try:
            with captured_socket(AF_UNIX, SOCK_STREAM) as canary:
                canary.connect("/tmp/sovereign-network-canary-does-not-exist")
        except NetworkDenied:
            canary_blocked = True

        if not canary_blocked or denied_events != ["socket.connect"]:
            raise AssertionError("audit guard did not block the socket alias canary")

        events_after_canary = len(denied_events)

        import asyncio
        import json
        from sovereign_api.contracts import ModelRequest
        from sovereign_api.providers import MockProvider

        response = asyncio.run(
            MockProvider().generate(
                ModelRequest(model_id="mock-fast", prompt="No network")
            )
        )
        if len(denied_events) != events_after_canary:
            raise AssertionError("MockProvider attempted network access")

        print(
            json.dumps(
                {
                    "canary_blocked": canary_blocked,
                    "provider_content": response.content,
                    "denied_events": denied_events,
                }
            )
        )
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result == {
        "canary_blocked": True,
        "provider_content": "Mock response from mock-fast",
        "denied_events": ["socket.connect"],
    }
